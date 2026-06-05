//! RPC module — JSON-RPC server for Python sidecar communication
//!
//! Provides a Unix domain socket JSON-RPC server that allows external
//! tools (primarily the Python sidecar) to interact with PincherOS.

pub mod server;

// Re-export key types
pub use server::{
    start_rpc_server, EngineCommand, JsonRpcRequest, JsonRpcResponse, RpcError, RpcErrorValue,
    RpcRequest, RpcResponse, RpcResult,
};
