import pytest
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool


def test_ipc_store_contract(tmp_path):
    tcp_store = dist.TCPStore(
        host_name="127.0.0.1",
        port=0,
        world_size=1,
        is_master=True,
        wait_for_workers=False,
    )
    file_store = dist.FileStore(str(tmp_path / "store"), world_size=1)
    nested_store = dist.PrefixStore(
        "outer", dist.PrefixStore("inner", file_store)
    )
    nested_tcp_store = dist.PrefixStore("outer", tcp_store)

    for store in (tcp_store, file_store, nested_store, nested_tcp_store):
        IPCBufferPool._assert_supported_ipc_store(store)
        store.set("aiter/ipc-store-contract", b"payload")
        assert store.get("aiter/ipc-store-contract") == b"payload"


def test_ipc_store_contract_rejects_unknown_store():
    class UnsupportedStore:
        pass

    with pytest.raises(TypeError, match="blocking KV store"):
        IPCBufferPool._assert_supported_ipc_store(UnsupportedStore())
