import json
import logging
import os
import warnings
from typing import List, Optional

from filelock import FileLock

# Must be set before importing databricks.sql — the pyarrow warning fires at import time
warnings.filterwarnings("ignore", message=".*pyarrow.*")

import urllib3
import pandas as pd
from databricks import sql

import config
from db.base import StorageBackend
from models import ProjectData

# Suppress urllib3 unverified-HTTPS warning (we use _tls_no_verify=True intentionally)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Silence verbose internal HTTP/session logs from the Databricks SQL connector
logging.getLogger("databricks.sql").setLevel(logging.ERROR)
logging.getLogger("databricks.sql.thrift_backend").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Maps doc_type values from Databricks to the two-bucket category used by the classifier prompt.
# This is a best-effort hint only — the classifier agent performs the real identification.
_DOC_TYPE_TO_CATEGORY = {
    "rera registration certificate": "rc_ec",
    "occupancy certificate":         "oc_cc",
    "completion certificate":        "oc_cc",
}

logger = logging.getLogger(__name__)


class DatabricksStorageBackend(StorageBackend):

    _table_ensured: bool = False  # class-level flag; DDL runs at most once per process

    def _get_connection(self):
        """Establish a connection to Databricks SQL warehouse."""
        try:
            server_hostname = config.DATABRICKS_SERVER_HOSTNAME
            http_path = config.DATABRICKS_HTTP_PATH
            access_token = config.DATABRICKS_ACCESS_TOKEN

            if not all([server_hostname, http_path, access_token]):
                raise ValueError(
                    "Missing required Databricks credentials. "
                    "Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, "
                    "and DATABRICKS_ACCESS_TOKEN in your environment."
                )

            return sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=access_token,
                _tls_no_verify=True,
                _retry_stop_after_attempts_count=3,
            )
        except Exception as e:
            logger.error(f"[Databricks] Failed to establish connection: {e}")
            raise

    # ------------------------------------------------------------------
    # get_document_urls
    # ------------------------------------------------------------------

    def get_document_urls(self, rera_id: str) -> Optional[List[dict]]:
        """
        Fetch document URLs for the given rera_id from the RERA xbyte document details table.

        Each row's `doc` column is a JSON array of {"url": ..., "name": ...} objects.
        The `doc_type` column is used as a category hint (rc_ec / oc_cc) but is NOT
        treated as ground truth — the classifier agent performs the real identification.

        Returns a list of {"url": str, "category": str} dicts ready for WorkflowState,
        or an empty list if no matching rows are found.
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            query = """
                SELECT doc_type, doc
                FROM   housing_np_production.rera_xbyte_document_details
                WHERE  rera_id = ?
                AND    doc_type IN (
                    'RERA Registration Certificate',
                    'Occupancy Certificate',
                    'Completion Certificate'
                )
            """
            cursor.execute(query, parameters=[rera_id])
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)

            if df.empty:
                logger.warning(f"[Databricks] get_document_urls() found no rows for rera_id={rera_id}")
                return []

            url_entries: List[dict] = []
            for _, row in df.iterrows():
                doc_type = str(row.get("doc_type") or "").strip()
                category = _DOC_TYPE_TO_CATEGORY.get(doc_type.lower(), "rc_ec")

                raw_doc = row.get("doc")
                if not raw_doc:
                    continue
                try:
                    doc_list = json.loads(raw_doc) if isinstance(raw_doc, str) else raw_doc
                    if not isinstance(doc_list, list):
                        doc_list = [doc_list]
                    for entry in doc_list:
                        url = entry.get("url", "").strip()
                        if url:
                            name = entry.get("name", "").strip() or url.split("/")[-1]
                            url_entries.append({"url": url, "name": name, "category": category})
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"[Databricks] Could not parse doc field for rera_id={rera_id}: {e}")

            # Keep rc_ec entries first, oc_cc entries after — same ordering as JSON mode.
            url_entries.sort(key=lambda e: 0 if e["category"] == "rc_ec" else 1)

            logger.info(f"[Databricks] get_document_urls() found {len(url_entries)} URL(s) for rera_id={rera_id}")
            return url_entries

        except Exception as e:
            logger.error(f"[Databricks] get_document_urls() failed for rera_id={rera_id}: {e}")
            raise
        finally:
            cursor.close()
            connection.close()

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    _TABLE = "data_science_metastore.rera_document_parser.extracted_rera_details"

    def _ensure_table(self, cursor) -> None:
        """Create the output table if it does not already exist. Runs at most once per process."""
        if DatabricksStorageBackend._table_ensured:
            return
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
                rera_id        STRING    NOT NULL,
                project_data   STRING,
                processed_docs STRING,
                last_updated   TIMESTAMP
            )
            USING DELTA
        """)
        DatabricksStorageBackend._table_ensured = True

    def load(self, rera_id: str) -> Optional[ProjectData]:
        """
        Fetch an existing ProjectData record for the given rera_id.
        Returns None if no record exists yet.
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            self._ensure_table(cursor)
            query = f"""
                SELECT project_data, processed_docs
                FROM   {self._TABLE}
                WHERE  rera_id = ?
                LIMIT  1
            """
            cursor.execute(query, parameters=[rera_id])
            rows = cursor.fetchall()
            if not rows:
                return None

            project_raw, processed_raw = rows[0][0], rows[0][1]
            if project_raw is None:
                return None

            try:
                data      = json.loads(project_raw)   if isinstance(project_raw,  str) else (project_raw  or {})
                processed = json.loads(processed_raw)  if isinstance(processed_raw, str) else (processed_raw or [])
            except json.JSONDecodeError as je:
                logger.warning(
                    f"[Databricks] load() found corrupted JSON for rera_id={rera_id}, "
                    f"treating as empty so fresh data can overwrite it: {je}"
                )
                return None

            data["processed_docs"] = processed
            return ProjectData(**data)

        except Exception as e:
            logger.error(f"[Databricks] load() failed for rera_id={rera_id}: {e}")
            raise
        finally:
            cursor.close()
            connection.close()

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(self, project: ProjectData) -> None:
        """
        Upsert a ProjectData record into the extracted_rera_details Delta table.
        When LOCAL_JSON_DUMP is enabled, also writes a local JSON file under STORAGE_DIR
        for inspection. The file is never read back by load() — Databricks is the source of truth.
        """
        if config.LOCAL_JSON_DUMP:
            os.makedirs(config.STORAGE_DIR, exist_ok=True)
            path = self._local_json_path(project.rera_id)
            with FileLock(path + ".lock"):
                with open(path, "w") as f:
                    f.write(project.model_dump_json(indent=2))
            logger.info(f"[Databricks] local JSON written to {path}")

        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            self._ensure_table(cursor)

            # Split processed_docs into its own column; project_data holds extracted output only.
            data = json.loads(project.model_dump_json())
            processed_docs_list = data.pop("processed_docs", [])
            project_json   = json.dumps(data)
            processed_json = json.dumps(processed_docs_list)

            # Use parameterized queries to avoid SQL-injection and broken escaping.
            # The naive .replace("'", "\\'") approach corrupted stored JSON, causing
            # JSONDecodeError on subsequent load() calls.
            query = f"""
                MERGE INTO {self._TABLE} AS target
                USING (SELECT ? AS rera_id) AS source
                ON target.rera_id = source.rera_id
                WHEN MATCHED THEN UPDATE SET
                    project_data   = ?,
                    processed_docs = ?,
                    last_updated   = current_timestamp()
                WHEN NOT MATCHED THEN INSERT (rera_id, project_data, processed_docs, last_updated)
                    VALUES (?, ?, ?, current_timestamp())
            """
            params = [project.rera_id, project_json, processed_json,
                      project.rera_id, project_json, processed_json]
            cursor.execute(query, parameters=params)
            logger.info(f"[Databricks] save() upserted rera_id={project.rera_id}")

        except Exception as e:
            logger.error(f"[Databricks] save() failed for rera_id={project.rera_id}: {e}")
            raise
        finally:
            cursor.close()
            connection.close()

    def _local_json_path(self, rera_id: str) -> str:
        safe_id = rera_id.replace("/", "_")
        return os.path.join(config.STORAGE_DIR, f"{safe_id}.json")

