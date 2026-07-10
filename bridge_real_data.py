import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_real_lnp_data():
    url = "https://raw.githubusercontent.com/evancollins1/LNPDB/main/data/LNPDB_for_LiON/single_split/test.csv"
    logger.info("[Pipeline] Streaming live formulation matrices from LNPDB...")
    df = pd.read_csv(url)
    real_smiles = df['smiles'].tolist()
    real_scores = df['target'].tolist()
    logger.info("Loaded %d real LNP formulations (SMILES + target)", len(real_smiles))
    return real_smiles, real_scores


def load_real_aav_fitness_data():
    url = "https://raw.githubusercontent.com/churchlab/AAV_fitness_landscape/main/data/processed/fitness_scores.csv"
    logger.info("[Pipeline] Streaming AAV fitness landscape from Church lab...")
    df = pd.read_csv(url)
    logger.info("Loaded AAV fitness landscape: %d variants, columns: %s", len(df), list(df.columns))
    return df


def load_fit4function_screening():
    url = "https://raw.githubusercontent.com/vector-engineering/fit4function/main/data/screening_results.csv"
    logger.info("[Pipeline] Streaming Fit4Function screening data...")
    try:
        df = pd.read_csv(url)
        logger.info("Loaded Fit4Function: %d records", len(df))
        return df
    except Exception as e:
        logger.warning("Fit4Function direct load failed: %s. Check the repo for available data files.", e)
        return None


if __name__ == "__main__":
    smiles, scores = load_real_lnp_data()
    logger.info("Sample SMILES: %s", smiles[:2])
    logger.info("Sample scores: %s", scores[:2])

    aav_df = load_real_aav_fitness_data()
    if aav_df is not None:
        logger.info("AAV columns: %s", list(aav_df.columns))

    f4f = load_fit4function_screening()
    if f4f is not None:
        logger.info("Fit4Function columns: %s", list(f4f.columns))
