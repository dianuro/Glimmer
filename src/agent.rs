use std::env;

use adk_model::OpenAICompatible;
use adk_model::OpenAICompatibleConfig;
use adk_rust::prelude::*;
use dotenvy::dotenv;

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

use adk_core::{Agent, Content, Memory, Part, Result, SessionId, UserId};
use adk_session::{CreateRequest, InMemorySessionService, SessionService};
use futures::StreamExt;

use adk_runner::{Runner, RunnerConfig};
pub async fn run() -> anyhow::Result<()> {
    dotenv().ok();

    let api_key = env::var("API_KEY")?;
    let model_id = env::var("MODEL_ID")?;
    let base_url = env::var("BASE_URL")?;

    let model = OpenAICompatible::new(
        OpenAICompatibleConfig::new(api_key, model_id)
            .with_base_url(base_url)
            .with_provider_name("custom"),
    )?;

    let agent = LlmAgentBuilder::new("assistant")
        .description("A helpful assistant")
        .instruction("You are a friendly assistant. Answer questions concisely.")
        .model(Arc::new(model))
        .build()?;

    Launcher::new(Arc::new(agent)).run().await?;
    Ok(())
}

pub struct Launcher {
    agent: Arc<dyn Agent>,
    app_name: Option<String>,
    session_service: Option<Arc<dyn SessionService>>,
    memory_service: Option<Arc<dyn Memory>>,
}

impl Launcher {
    /// Create a new launcher with the given agent.
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        Self {
            agent,
            app_name: None,
            session_service: None,
            memory_service: None,
        }
    }

    /// Set a custom application name (defaults to agent name).
    pub fn app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = Some(name.into());
        self
    }

    /// Set a custom session service (defaults to in-memory).
    pub fn with_session_service(mut self, service: Arc<dyn SessionService>) -> Self {
        self.session_service = Some(service);
        self
    }

    /// Set a custom memory service.
    pub fn with_memory_service(mut self, service: Arc<dyn Memory>) -> Self {
        self.memory_service = Some(service);
        self
    }

    /// Run the interactive console loop.
    ///
    /// Reads lines from stdin, sends them to the agent, and prints streaming
    /// responses. Type `exit` or `quit` to stop. Ctrl+C exits immediately.
    pub async fn run(self) -> Result<()> {
        let app_name = self
            .app_name
            .unwrap_or_else(|| self.agent.name().to_string());
        let agent = self.agent;

        let session_service: Arc<dyn SessionService> = self
            .session_service
            .unwrap_or_else(|| Arc::new(InMemorySessionService::new()));

        let session = session_service
            .create(CreateRequest {
                app_name: app_name.clone(),
                user_id: "user".into(),
                session_id: None,
                state: HashMap::new(),
            })
            .await?;

        let session_id = session.id().to_string();

        println!();
        println!("  Glimmer⭐ — {}", agent.name());
        println!("  Type a message to chat.");
        println!();

        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        loop {
            print!("You > ");
            io::stdout().flush().ok();

            let line = match lines.next() {
                Some(Ok(line)) => line,
                Some(Err(_)) | None => break,
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if matches!(trimmed, "exit" | "quit" | "/exit" | "/quit") {
                println!("\nGoodbye.\n");
                break;
            }

            let user_content = Content::new("user").with_text(trimmed);

            let runner = Runner::new(RunnerConfig {
                app_name: app_name.clone(),
                agent: agent.clone(),
                session_service: session_service.clone(),
                artifact_service: None,
                memory_service: self.memory_service.clone(),
                run_config: None,
                compaction_config: None,
                context_cache_config: None,
                cache_capable: None,
                request_context: None,
                cancellation_token: None,
                intra_compaction_config: None,
                intra_compaction_summarizer: None,
                plugin_manager: None,
            })?;

            let mut stream = runner
                .run(
                    UserId::new("user")?,
                    SessionId::new(&session_id)?,
                    user_content,
                )
                .await?;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(evt) => {
                        if let Some(content) = evt.content() {
                            for part in &content.parts {
                                match part {
                                    Part::Text { text } => {
                                        print!("{text}");
                                        io::stdout().flush().ok();
                                    }
                                    Part::Thinking { thinking, .. } => {
                                        print!("\n[thinking] {thinking}");
                                        io::stdout().flush().ok();
                                    }
                                    Part::FunctionCall { name, args, .. } => {
                                        print!("\n[tool] {name}({args})");
                                        io::stdout().flush().ok();
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("\n[error] {e}");
                    }
                }
            }

            println!("\n");
        }

        Ok(())
    }
}
