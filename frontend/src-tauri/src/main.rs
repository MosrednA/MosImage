#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engine;

use tauri::{Manager, RunEvent};

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(engine::Engine::start());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<engine::Engine>().stop();
            }
        });
}
