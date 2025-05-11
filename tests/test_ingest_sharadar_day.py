import os
import sys
import pandas as pd
import pytest

env = os.environ.copy()
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ingest_sharadar_day as ish

@pytest.fixture(autouse=True)
def setup_tmpdir(tmp_path, monkeypatch):
    """Redirect SEP_DIR to a temporary directory and set dummy API key."""
    sep_dir = tmp_path / "sep_dataset"
    sep_dir.mkdir()
    monkeypatch.setattr(ish, 'SEP_DIR', str(sep_dir))
    monkeypatch.setenv('NASDAQ_API_KEY', 'DUMMY_KEY')
    return sep_dir

def write_parquet(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def test_bootstrap_full_history(setup_tmpdir, monkeypatch):
    """On first run, ingest_table should bootstrap full history from SRC."""
    sep_dir = setup_tmpdir
    # Create a small source SEP file
    df_src = pd.DataFrame({
        'ticker': ['AAA', 'BBB'],
        'date': ['2025-01-02', '2025-01-02'],
        'volume': [100, 200]
    })
    df_src['date'] = pd.to_datetime(df_src['date'])
    src_path = sep_dir / 'SHARADAR_SEP_2.parquet'
    write_parquet(src_path, df_src)

    # Monkeypatch API to return empty DataFrame
    monkeypatch.setattr(ish.nasdaqdatalink, 'get_table', lambda *args, **kwargs: pd.DataFrame())

    # Call ingest_table for bootstrap
    target_date = pd.to_datetime('2025-01-02').date()
    ish.ingest_table(target_date, 'SEP', 'SHARADAR_SEP_2.parquet', 'SHARADAR_SEP.parquet', 'date', ['ticker','date'])

    # Master file should now exist and equal df_src
    master_df = pd.read_parquet(sep_dir / 'SHARADAR_SEP.parquet')
    master_df['date'] = pd.to_datetime(master_df['date'])
    # Compare ignoring index
    pd.testing.assert_frame_equal(
        master_df.sort_values(['ticker','date']).reset_index(drop=True),
        df_src.sort_values(['ticker','date']).reset_index(drop=True)
    )


def test_incremental_append(setup_tmpdir, monkeypatch):
    """After bootstrap, ingest_table should append only new rows."""
    sep_dir = setup_tmpdir
    # Bootstrap first
    df_full = pd.DataFrame({
        'ticker': ['AAA'],
        'date': [pd.to_datetime('2025-01-02')],
        'volume': [100]
    })
    write_parquet(sep_dir / 'SHARADAR_SEP.parquet', df_full)

    # Create a source with an existing and a new date
    df_src = pd.DataFrame({
        'ticker': ['AAA','AAA'],
        'date': ['2025-01-02','2025-01-03'],
        'volume': [100,150]
    })
    df_src['date'] = pd.to_datetime(df_src['date'])
    write_parquet(sep_dir / 'SHARADAR_SEP_2.parquet', df_src)

    monkeypatch.setattr(ish.nasdaqdatalink, 'get_table', lambda *args, **kwargs: pd.DataFrame())

    # Append date 2025-01-03
    target_date = pd.to_datetime('2025-01-03').date()
    ish.ingest_table(target_date, 'SEP', 'SHARADAR_SEP_2.parquet', 'SHARADAR_SEP.parquet', 'date', ['ticker','date'])

    # Master should now have both dates
    master_df = pd.read_parquet(sep_dir / 'SHARADAR_SEP.parquet')
    master_df['date'] = pd.to_datetime(master_df['date'])
    assert set(master_df['date'].dt.date) == {pd.to_datetime('2025-01-02').date(), pd.to_datetime('2025-01-03').date()}
    # Volume for new date present
    assert master_df[master_df['date'].dt.date == pd.to_datetime('2025-01-03').date()]['volume'].iloc[0] == 150
