#!/usr/bin/env python3
"""Upload benchmark WAV audio rows to Google Drive and annotate report JSON."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SDK_FOLDERS = ("Flutter SDK", "Web SDK", "Swift SDK", "RN SDK", "Python SDK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--root-folder-id", default=os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID"))
    parser.add_argument("--sdk-folder", default=os.getenv("TESTMU_SDK_FOLDER", "RN SDK"))
    parser.add_argument("--pr-number", default=os.getenv("GITHUB_EVENT_NUMBER"))
    parser.add_argument("--pr-folder", default=os.getenv("TESTMU_DRIVE_PR_FOLDER"))
    parser.add_argument("--public", action="store_true", default=is_truthy(os.getenv("GOOGLE_DRIVE_PUBLIC_AUDIO")))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def read_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def slugify(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or "unknown"


def drive_link(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def folder_link(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def service_account_info() -> dict[str, Any] | None:
    json_value = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if json_value:
        return json.loads(json_value)

    json_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_FILE")
    if json_path:
        return json.loads(Path(json_path).read_text())

    return None


def build_drive_service():
    info = service_account_info()
    if not info:
        return None

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def ensure_folder(service, parent_id: str, name: str) -> dict[str, str]:
    query = (
        f"mimeType = '{FOLDER_MIME_TYPE}' and "
        f"name = '{query_literal(name)}' and "
        f"'{query_literal(parent_id)}' in parents and trashed = false"
    )
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name,webViewLink)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = response.get("files") or []
    if files:
        return files[0]

    return (
        service.files()
        .create(
            body={"name": name, "mimeType": FOLDER_MIME_TYPE, "parents": [parent_id]},
            fields="id,name,webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def find_file(service, parent_id: str, name: str) -> dict[str, str] | None:
    query = (
        f"name = '{query_literal(name)}' and "
        f"'{query_literal(parent_id)}' in parents and trashed = false"
    )
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name,webViewLink)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = response.get("files") or []
    return files[0] if files else None


def make_public_reader(service, file_id: str) -> None:
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
        fields="id",
        supportsAllDrives=True,
    ).execute()


def upload_wav(service, folder_id: str, name: str, wav_bytes: bytes, public: bool) -> dict[str, str]:
    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(io.BytesIO(wav_bytes), mimetype="audio/wav", resumable=False)
    existing = find_file(service, folder_id, name)
    if existing:
        file_info = (
            service.files()
            .update(
                fileId=existing["id"],
                media_body=media,
                fields="id,name,webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )
    else:
        file_info = (
            service.files()
            .create(
                body={"name": name, "parents": [folder_id], "mimeType": "audio/wav"},
                media_body=media,
                fields="id,name,webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )

    if public:
        make_public_reader(service, file_info["id"])

    return file_info


def pr_folder_name(args: argparse.Namespace) -> str:
    if args.pr_folder:
        return args.pr_folder
    if args.pr_number:
        return f"PR #{int(args.pr_number):03d}"
    return "PR #local"


def annotate_row_skipped(row: dict[str, Any], message: str) -> None:
    row["audioUploadStatus"] = "skipped"
    row["audioUploadErrorSummary"] = message


def annotate_report_upload(report: dict[str, Any], status: str, folder_id: str | None, counts: dict[str, int], error: str | None = None) -> None:
    report["audioUpload"] = {
        "provider": "google-drive",
        "status": status,
        "folderId": folder_id,
        "folderUrl": folder_link(folder_id) if folder_id else None,
        "uploadedRows": counts.get("uploaded", 0),
        "skippedRows": counts.get("skipped", 0),
        "failedRows": counts.get("failed", 0),
        "errorSummary": error,
    }


def friendly_error(error: Exception) -> str:
    message = str(error)
    if "Service Accounts do not have storage quota" in message:
        return (
            "Google Drive rejected the WAV upload because service accounts do not "
            "have storage quota. Use a Google Shared Drive folder or OAuth "
            "delegation for Drive audio uploads."
        )
    return message


def upload_reports(args: argparse.Namespace) -> None:
    files = read_json_files(args.input_dir)
    if not files:
        return

    service = None if args.dry_run else build_drive_service()
    setup_error = None
    if not args.dry_run and not service:
        setup_error = "Google Drive upload is not configured."
    if not args.dry_run and not args.root_folder_id:
        setup_error = "GOOGLE_DRIVE_ROOT_FOLDER_ID is not configured."

    pr_name = pr_folder_name(args)
    pr_folder_id = None
    if args.dry_run:
        pr_folder_id = "dry-run-pr-folder"
    elif not setup_error:
        try:
            for folder_name in SDK_FOLDERS:
                ensure_folder(service, args.root_folder_id, folder_name)
            sdk_folder = ensure_folder(service, args.root_folder_id, args.sdk_folder)
            pr_folder = ensure_folder(service, sdk_folder["id"], pr_name)
            pr_folder_id = pr_folder["id"]
        except Exception as error:  # noqa: BLE001
            setup_error = f"Could not prepare Google Drive folders: {friendly_error(error)}"

    for file_path in files:
        report = json.loads(file_path.read_text())
        rows = report.get("rows") if isinstance(report.get("rows"), list) else []
        counts = {"uploaded": 0, "skipped": 0, "failed": 0}

        for row in rows:
            if row.get("status") != "passed":
                continue
            if not row.get("werAudioBase64"):
                annotate_row_skipped(row, "Benchmark row did not include WAV audio.")
                counts["skipped"] += 1
                continue
            if setup_error:
                annotate_row_skipped(row, setup_error)
                counts["skipped"] += 1
                continue

            file_name = "__".join(
                [
                    slugify(report.get("device")),
                    slugify(row.get("modelDisplayName") or row.get("model")),
                    slugify(row.get("sampleHash")),
                ]
            ) + ".wav"
            row["audioFileName"] = file_name

            try:
                wav_bytes = base64.b64decode(row["werAudioBase64"])
                if args.dry_run:
                    file_id = f"dry-run-{slugify(report.get('device'))}-{slugify(row.get('model'))}"
                    link = drive_link(file_id)
                else:
                    uploaded = upload_wav(service, pr_folder_id, file_name, wav_bytes, args.public)
                    file_id = uploaded["id"]
                    link = uploaded.get("webViewLink") or drive_link(file_id)

                row["audioUploadStatus"] = "passed"
                row["audioDriveFileId"] = file_id
                row["audioListenUrl"] = link
                row.pop("audioUploadErrorSummary", None)
                counts["uploaded"] += 1
            except Exception as error:  # noqa: BLE001
                row["audioUploadStatus"] = "failed"
                row["audioUploadErrorSummary"] = friendly_error(error)
                counts["failed"] += 1

        status = (
            "passed"
            if counts["uploaded"] and not counts["failed"]
            else "failed"
            if counts["failed"]
            else "skipped"
            if counts["skipped"]
            else "unavailable"
        )
        annotate_report_upload(report, status, pr_folder_id, counts, setup_error)
        file_path.write_text(f"{json.dumps(report, indent=2)}\n")


if __name__ == "__main__":
    upload_reports(parse_args())
