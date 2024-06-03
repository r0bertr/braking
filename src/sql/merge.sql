-- Attach the second database
ATTACH DATABASE 'probe_data_annotated.db' AS b;

CREATE TABLE IF NOT EXISTS "probe_data"(
  "csv_identifier" TEXT,
  "seq_name" TEXT,
  "datetime" TIMESTAMP,
  "speed" INTEGER,
  "accel_x" REAL,
  "accel_y" REAL,
  "accel_z" REAL,
  "accel" REAL,
  "latitude" REAL,
  "longitude" REAL,
  "direction" REAL,
  "braking_flag" INTEGER,
  "braking_description" TEXT default "",
  delta_dist_mean_3secs REAL default 0,
  dist_mean_3secs REAL default -1,
  n_dets_3secs REAL default 0,
  dist_mean_1secs REAL default -1,
  delta_dist_mean_1secs REAL default 0,
  n_dets_1secs REAL default 0
);
CREATE INDEX IF NOT EXISTS "ix_probe_data_csv_identifier_seq_name" ON "probe_data" (
  "csv_identifier",
  "seq_name"
);
CREATE INDEX IF NOT EXISTS "ix_probe_data_csv_identifier_seq_name_datetime" ON "probe_data" (
  "csv_identifier",
  "seq_name",
  "datetime"
);
CREATE INDEX IF NOT EXISTS "ix_probe_data_braking_flag" ON "probe_data"("braking_flag");
CREATE INDEX IF NOT EXISTS "ix_probe_data_accel_x" ON "probe_data"("accel_x");
CREATE INDEX IF NOT EXISTS "ix_probe_data_csv_identifier_seq_name_braking_flag" ON "probe_data"(
  "csv_identifier",
  "seq_name",
  "braking_flag"
);

BEGIN TRANSACTION;
-- Create temporary table
CREATE TEMPORARY TABLE temp_ids (csv_identifier TEXT, seq_name TEXT);

-- Import IDs from the text file
.mode csv
.import seq_names.csv temp_ids

INSERT INTO probe_data (
    "csv_identifier",
    "seq_name",
    "datetime",
    "speed",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel",
    "latitude",
    "longitude",
    "direction",
    "braking_flag",
    "braking_description",
    "delta_dist_mean_3secs",
    "dist_mean_3secs",
    "n_dets_3secs",
    "dist_mean_1secs",
    "delta_dist_mean_1secs",
    "n_dets_1secs"
)
SELECT "csv_identifier","seq_name","datetime","speed","accel_x","accel_y","accel_z","accel","latitude","longitude","direction","braking_flag","braking_description","delta_dist_mean_3secs","dist_mean_3secs","n_dets_3secs","dist_mean_1secs","delta_dist_mean_1secs","n_dets_1secs"
FROM b.probe_data WHERE EXISTS (SELECT 1 FROM temp_ids WHERE b.probe_data.csv_identifier = temp_ids.csv_identifier AND b.probe_data.seq_name = temp_ids.seq_name);

COMMIT;

-- Detach the second database
DETACH DATABASE b;
