import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import data.preprocess_data as ppd
import features.build_features as bf
import pandas as pd

EXCLUDED_COLUMNS = [
    "falha_24h",
    "data_registro_pad",
]

FLOAT_COLUMNS = [
    "paradas_nao_planejadas",
    "idade_maquina_anos",
    "temperatura_c",
    "pressao_bar",
    "vibracao_motor_mm_s",
    "velocidade_esteira_m_min",
    "umidade_pct",
    "tamanho_lote",
    "tempo_setup_min",
    "taxa_defeitos_pct",
    "linha_producao_pad_A",
    "linha_producao_pad_B",
    "linha_producao_pad_C",
    "turno_pad_manha",
    "turno_pad_noite",
    "turno_pad_tarde",
    "maquina_M01",
    "maquina_M02",
    "maquina_M03",
    "maquina_M04",
    "maquina_M05",
    "maquina_M06",
    "maquina_M07",
    "maquina_M08",
    "consumo_energia_kwh",
]


def normalize_output_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the transformed frame to the serving schema."""
    df = df.copy()

    for column in FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="raise").astype("float64")

    df = df.drop(
        columns=[column for column in EXCLUDED_COLUMNS if column in df.columns]
    )

    return df


def main(df = None , input_dict=None ):

    # Gerate and create dataframe for requisition
    if df is None:
        df = pd.DataFrame([input_dict])
    
    # Preprocess data
    df = ppd.main(df)

    # Build features
    df = bf.main(df)

    return normalize_output_types(df)




if __name__ == "__main__":
    main()