#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def parse_elf_sections(data):
    section_header_offset = struct.unpack_from("<Q", data, 0x28)[0]
    section_header_size = struct.unpack_from("<H", data, 0x3A)[0]
    section_count = struct.unpack_from("<H", data, 0x3C)[0]
    section_name_table_index = struct.unpack_from("<H", data, 0x3E)[0]
    headers = []
    for index in range(section_count):
        offset = section_header_offset + index * section_header_size
        name_offset, section_type = struct.unpack_from("<II", data, offset)
        _, _, section_offset, section_size = struct.unpack_from("<QQQQ", data, offset + 8)
        headers.append((name_offset, section_type, section_offset, section_size))
    name_table_header = headers[section_name_table_index]
    name_table = data[name_table_header[2] : name_table_header[2] + name_table_header[3]]
    sections = {}
    for header in headers:
        name_end = name_table.find(b"\0", header[0])
        name = name_table[header[0] : name_end].decode()
        if name:
            sections[name] = data[header[2] : header[2] + header[3]]
    return sections


def read_json(path):
    return json.loads(path.read_text())


def final_builder_information(scenario_dir):
    events = []
    with (scenario_dir / "events.jsonl").open() as events_file:
        for line in events_file:
            event = json.loads(line)
            if event["event"] == "compile_end":
                events.append(event)
    if not events:
        return None, None
    final_rank = events[-1]["rank"]
    processes = read_json(scenario_dir / "processes.json")
    for process in processes:
        if process["rank"] == final_rank:
            return final_rank, process["pythonhashseed"]
    return final_rank, None


def normalized_compile_flags(makefile_path):
    for line in makefile_path.read_text().splitlines():
        if "$(CXX) -fPIC" not in line:
            continue
        flags = line.split("$(CXX) ", 1)[1]
        flags = re.sub(r"-I\S+", "", flags)
        return flags.strip()
    raise ValueError(f"compile rule not found: {makefile_path}")


def analyze_scenario(scenario_dir):
    artifacts = read_json(scenario_dir / "artifacts.json")
    artifact = artifacts[0]
    library_path = Path(artifact["path"])
    library_data = library_path.read_bytes()
    host_sections = parse_elf_sections(library_data)
    fatbin = host_sections[".hip_fatbin"]
    embedded_offset = fatbin.find(b"\x7fELF")
    embedded_elf = fatbin[embedded_offset:]
    device_sections = parse_elf_sections(embedded_elf)
    cuid_match = re.search(rb"__hip_cuid_[0-9a-f]+", library_data)
    final_builder_rank, final_builder_seed = final_builder_information(scenario_dir)
    flags = normalized_compile_flags(library_path.parent / "Makefile")
    return {
        "scenario": scenario_dir.name,
        "library_path": str(library_path),
        "library_size": artifact["size"],
        "whole_library_sha256": artifact["sha256"],
        "host_text_size": len(host_sections[".text"]),
        "host_text_sha256": sha256_bytes(host_sections[".text"]),
        "hip_fatbin_sha256": sha256_bytes(fatbin),
        "embedded_elf_offset": embedded_offset,
        "device_text_size": len(device_sections[".text"]),
        "device_text_sha256": sha256_bytes(device_sections[".text"]),
        "device_note_size": len(device_sections[".note"]),
        "device_note_sha256": sha256_bytes(device_sections[".note"]),
        "hip_cuid": cuid_match.group(0).decode() if cuid_match else None,
        "final_builder_rank": final_builder_rank,
        "final_builder_pythonhashseed": final_builder_seed,
        "normalized_compile_flags": flags,
        "normalized_compile_flags_sha256": sha256_bytes(flags.encode()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for scenario_dir in sorted(args.matrix_root.iterdir()):
        if not (scenario_dir / "artifacts.json").exists():
            continue
        records.append(analyze_scenario(scenario_dir))
    summary = {
        "records": records,
        "unique_whole_library_hashes": sorted({record["whole_library_sha256"] for record in records}),
        "unique_host_text_hashes": sorted({record["host_text_sha256"] for record in records}),
        "unique_device_text_hashes": sorted({record["device_text_sha256"] for record in records}),
        "unique_device_note_hashes": sorted({record["device_note_sha256"] for record in records}),
        "unique_normalized_flag_orders": sorted({record["normalized_compile_flags_sha256"] for record in records}),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
