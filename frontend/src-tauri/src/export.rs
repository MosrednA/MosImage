//! Saves an exported picture where the user picks, instead of the webview's silent download to Downloads.

use std::path::Path;

use percent_encoding::percent_decode_str;
use tauri::ipc::{InvokeBody, Request};
use tauri::WebviewWindow;
use tauri_plugin_dialog::DialogExt;

/// Takes the image bytes as the raw request body and the suggested name in the `x-file-name` header (URI-encoded).
/// Returns the saved path, or `None` when the user cancels the dialog.
#[tauri::command]
pub async fn save_image(window: WebviewWindow, request: Request<'_>) -> Result<Option<String>, String> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("Expected the image as raw bytes".into());
    };
    let name = request
        .headers()
        .get("x-file-name")
        .and_then(|value| value.to_str().ok())
        .map(|value| percent_decode_str(value).decode_utf8_lossy().into_owned())
        .unwrap_or_else(|| "mosimage.png".into());
    let extension = Path::new(&name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("png")
        .to_ascii_lowercase();
    let label = match extension.as_str() { "jpg" | "jpeg" => "JPEG image", "webp" => "WebP image", _ => "PNG image" };
    let Some(target) = window
        .dialog()
        .file()
        .set_parent(&window)
        .set_title("Export image")
        .set_file_name(&name)
        .add_filter(label, &[extension.as_str()])
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let path = target.into_path().map_err(|error| error.to_string())?;
    std::fs::write(&path, bytes).map_err(|error| format!("Could not save {}: {error}", path.display()))?;
    Ok(Some(path.display().to_string()))
}
