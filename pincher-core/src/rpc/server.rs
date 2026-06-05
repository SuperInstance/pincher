//! UDS JSON-RPC server for Python sidecar communication

use crate::reflex::engine::{EngineStatus, ReflexEngine};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tracing::{error, info, instrument, warn};

/// RPC server errors.
#[derive(Debug, Error)]
pub enum RpcError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON-RPC error: {0}")]
    JsonRpc(String),

    #[error("Engine error: {0}")]
    Engine(String),

    #[error("Channel closed")]
    ChannelClosed,
}

/// Result type for RPC operations.
pub type RpcResult<T> = Result<T, RpcError>;

/// RPC method request types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RpcRequest {
    Ping,
    EmbedText { text: String },
    MatchReflex { intent: String },
    TeachReflex { intent: String, action: String },
    GetStatus,
}

/// RPC method response types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RpcResponse {
    Ping {
        pong: String,
    },
    EmbedText {
        embedding: Vec<f32>,
        dimensions: usize,
    },
    MatchReflex {
        match_type: String,
        similarity: f32,
        reflex_id: Option<String>,
    },
    TeachReflex {
        reflex_id: String,
        intent: String,
        confidence: f64,
    },
    GetStatus {
        status: EngineStatus,
    },
}

/// JSON-RPC request structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC response structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcErrorValue>,
}

/// JSON-RPC error value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcErrorValue {
    pub code: i64,
    pub message: String,
}

/// Command sent from the RPC server to the engine.
#[derive(Debug)]
pub enum EngineCommand {
    EmbedText {
        text: String,
        response_tx: mpsc::Sender<RpcResponse>,
    },
    MatchReflex {
        intent: String,
        response_tx: mpsc::Sender<RpcResponse>,
    },
    TeachReflex {
        intent: String,
        action: String,
        response_tx: mpsc::Sender<RpcResponse>,
    },
    GetStatus {
        response_tx: mpsc::Sender<RpcResponse>,
    },
}

/// Start the JSON-RPC server on a Unix domain socket.
#[instrument(skip(socket_path, engine))]
pub async fn start_rpc_server(socket_path: &Path, engine: ReflexEngine) -> RpcResult<()> {
    info!(socket = ?socket_path, "Starting JSON-RPC server");

    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }

    let (cmd_tx, mut cmd_rx) = mpsc::channel::<EngineCommand>(100);

    let engine_handle = tokio::spawn(async move {
        let mut engine = engine;
        while let Some(cmd) = cmd_rx.recv().await {
            match cmd {
                EngineCommand::EmbedText { text, response_tx } => {
                    let result = engine.embedder().embed(&text);
                    let response = match result {
                        Ok(embedding) => RpcResponse::EmbedText {
                            dimensions: embedding.len(),
                            embedding,
                        },
                        Err(e) => {
                            warn!(error = %e, "Failed to embed text");
                            RpcResponse::EmbedText {
                                dimensions: 0,
                                embedding: vec![],
                            }
                        }
                    };
                    let _ = response_tx.send(response).await;
                }

                EngineCommand::MatchReflex {
                    intent,
                    response_tx,
                } => {
                    let result = crate::reflex::matcher::match_reflex(
                        engine.connection(),
                        engine.embedder(),
                        &intent,
                    );

                    let response = match result {
                        Ok(match_result) => {
                            let (match_type, similarity, reflex_id) = match &match_result {
                                crate::reflex::matcher::MatchResult::Exact {
                                    similarity,
                                    reflex,
                                } => ("exact".to_string(), *similarity, Some(reflex.id.clone())),
                                crate::reflex::matcher::MatchResult::Similar {
                                    similarity,
                                    reflex,
                                } => ("similar".to_string(), *similarity, Some(reflex.id.clone())),
                                crate::reflex::matcher::MatchResult::Novel { best_similarity } => {
                                    ("novel".to_string(), *best_similarity, None)
                                }
                            };
                            RpcResponse::MatchReflex {
                                match_type,
                                similarity,
                                reflex_id,
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to match reflex");
                            RpcResponse::MatchReflex {
                                match_type: "error".to_string(),
                                similarity: 0.0,
                                reflex_id: None,
                            }
                        }
                    };

                    let _ = response_tx.send(response).await;
                }

                EngineCommand::TeachReflex {
                    intent,
                    action,
                    response_tx,
                } => {
                    let result = engine.teach(&intent, &action);
                    let response = match result {
                        Ok(reflex) => RpcResponse::TeachReflex {
                            reflex_id: reflex.id,
                            intent: reflex.intent,
                            confidence: reflex.confidence,
                        },
                        Err(e) => {
                            warn!(error = %e, "Failed to teach reflex");
                            RpcResponse::TeachReflex {
                                reflex_id: String::new(),
                                intent: String::new(),
                                confidence: 0.0,
                            }
                        }
                    };

                    let _ = response_tx.send(response).await;
                }

                EngineCommand::GetStatus { response_tx } => {
                    let result = engine.get_status();
                    let response = match result {
                        Ok(status) => RpcResponse::GetStatus { status },
                        Err(e) => {
                            warn!(error = %e, "Failed to get status");
                            RpcResponse::GetStatus {
                                status: EngineStatus {
                                    reflex_count: 0,
                                    action_log_count: 0,
                                    embedder_loaded: false,
                                },
                            }
                        }
                    };

                    let _ = response_tx.send(response).await;
                }
            }
        }
    });

    let listener = UnixListener::bind(socket_path)?;

    // SECURITY: Restrict socket to owner-only (mode 600)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) =
            std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
        {
            warn!(error = %e, "Failed to set restrictive permissions on RPC socket");
        }
    }
    let cmd_tx_clone = cmd_tx;

    let socket_handle = tokio::spawn(async move {
        info!("RPC server listening on socket");

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    info!("New RPC client connected");
                    let cmd_tx = cmd_tx_clone.clone();

                    tokio::spawn(async move {
                        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

                        let (reader, mut writer) = stream.into_split();
                        let mut reader = BufReader::new(reader);
                        let mut line = String::new();

                        loop {
                            line.clear();
                            match reader.read_line(&mut line).await {
                                Ok(0) => break,
                                Ok(_) => {
                                    let request: JsonRpcRequest =
                                        match serde_json::from_str(line.trim()) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                let error_response = JsonRpcResponse {
                                                    jsonrpc: "2.0".to_string(),
                                                    id: None,
                                                    result: None,
                                                    error: Some(RpcErrorValue {
                                                        code: -32700,
                                                        message: format!("Parse error: {}", e),
                                                    }),
                                                };
                                                let response_str =
                                                    serde_json::to_string(&error_response)
                                                        .unwrap_or_default();
                                                let _ = writer
                                                    .write_all(
                                                        format!("{}\n", response_str).as_bytes(),
                                                    )
                                                    .await;
                                                continue;
                                            }
                                        };

                                    let response = handle_request(request, &cmd_tx).await;

                                    let response_str =
                                        serde_json::to_string(&response).unwrap_or_default();
                                    if let Err(e) = writer
                                        .write_all(format!("{}\n", response_str).as_bytes())
                                        .await
                                    {
                                        warn!(error = %e, "Failed to write RPC response");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "Failed to read from RPC client");
                                    break;
                                }
                            }
                        }
                    });
                }
                Err(e) => {
                    error!(error = %e, "Failed to accept RPC connection");
                }
            }
        }
    });

    info!("RPC server started successfully");

    tokio::select! {
        _ = engine_handle => {
            warn!("Engine handler task exited");
        }
        _ = socket_handle => {
            warn!("Socket listener task exited");
        }
    }

    Ok(())
}

/// Handle a single JSON-RPC request.
async fn handle_request(
    request: JsonRpcRequest,
    cmd_tx: &mpsc::Sender<EngineCommand>,
) -> JsonRpcResponse {
    let id = request.id.clone();

    match request.method.as_str() {
        "ping" => {
            let response = RpcResponse::Ping {
                pong: "pong".to_string(),
            };
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(serde_json::to_value(&response).unwrap_or_default()),
                error: None,
            }
        }

        "embed_text" => {
            let text = request
                .params
                .as_ref()
                .and_then(|p| p.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let (response_tx, mut response_rx) = mpsc::channel(1);

            if cmd_tx
                .send(EngineCommand::EmbedText { text, response_tx })
                .await
                .is_err()
            {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "Engine not available".to_string(),
                    }),
                };
            }

            match response_rx.recv().await {
                Some(response) => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::to_value(&response).unwrap_or_default()),
                    error: None,
                },
                None => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "No response from engine".to_string(),
                    }),
                },
            }
        }

        "match_reflex" => {
            let intent = request
                .params
                .as_ref()
                .and_then(|p| p.get("intent"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let (response_tx, mut response_rx) = mpsc::channel(1);

            if cmd_tx
                .send(EngineCommand::MatchReflex {
                    intent,
                    response_tx,
                })
                .await
                .is_err()
            {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "Engine not available".to_string(),
                    }),
                };
            }

            match response_rx.recv().await {
                Some(response) => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::to_value(&response).unwrap_or_default()),
                    error: None,
                },
                None => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "No response from engine".to_string(),
                    }),
                },
            }
        }

        "teach_reflex" => {
            let intent = request
                .params
                .as_ref()
                .and_then(|p| p.get("intent"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let action = request
                .params
                .as_ref()
                .and_then(|p| p.get("action"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let (response_tx, mut response_rx) = mpsc::channel(1);

            if cmd_tx
                .send(EngineCommand::TeachReflex {
                    intent,
                    action,
                    response_tx,
                })
                .await
                .is_err()
            {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "Engine not available".to_string(),
                    }),
                };
            }

            match response_rx.recv().await {
                Some(response) => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::to_value(&response).unwrap_or_default()),
                    error: None,
                },
                None => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "No response from engine".to_string(),
                    }),
                },
            }
        }

        "get_status" => {
            let (response_tx, mut response_rx) = mpsc::channel(1);

            if cmd_tx
                .send(EngineCommand::GetStatus { response_tx })
                .await
                .is_err()
            {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "Engine not available".to_string(),
                    }),
                };
            }

            match response_rx.recv().await {
                Some(response) => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::to_value(&response).unwrap_or_default()),
                    error: None,
                },
                None => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(RpcErrorValue {
                        code: -32603,
                        message: "No response from engine".to_string(),
                    }),
                },
            }
        }

        _ => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(RpcErrorValue {
                code: -32601,
                message: format!("Method not found: {}", request.method),
            }),
        },
    }
}
