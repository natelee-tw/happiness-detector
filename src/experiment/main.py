import logging
from polyaxon_client.tracking import Experiment, get_log_level, get_outputs_path

from src.modelling import model

logger = logging.getLogger(__name__)


def run_experiment(params):
    try:
        log_level = get_log_level()
        if not log_level:
            log_level = logging.INFO

        logger.info("Starting experiment")

        experiment = Experiment()
        logging.basicConfig(level=log_level)

        metrics = model.train(params)

        experiment.log_metrics(**metrics)

        logger.info("Experiment completed")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")


if __name__ == '__main__':
    in_params = {}  # Add your own params
    run_experiment(in_params)
