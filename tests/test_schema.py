import pandas as pd
import pandera.pandas as pa
import pytest

from src.mlops_project.data.load import load_bank_data
from src.mlops_project.data.schema import validate_bank_data


@pytest.fixture
def df() -> pd.DataFrame:
    return load_bank_data()


def test_donnees_valides(df: pd.DataFrame) -> None:
    validate_bank_data(df)


def test_colonne_renommee(df: pd.DataFrame) -> None:
    casse = df.rename(columns={"balance": "solde"})
    with pytest.raises(pa.errors.SchemaErrors):
        validate_bank_data(casse)


def test_age_negatif(df: pd.DataFrame) -> None:
    casse = df.copy()
    casse.loc[0, "age"] = -5
    with pytest.raises(pa.errors.SchemaErrors):
        validate_bank_data(casse)


def test_cible_inconnue(df: pd.DataFrame) -> None:
    casse = df.copy()
    casse.loc[0, "y"]  = "peut-etre"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_bank_data(casse)
