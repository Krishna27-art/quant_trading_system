import logging

from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobScheduler:
    """
    Central scheduler for the quant research platform.
    Manages daily data ingestion, feature generation, prediction, and retraining.
    """

    def __init__(self):
        self.scheduler = BackgroundScheduler()

    def start(self):
        self.scheduler.start()
        logger.info("Scheduler started.")

    def shutdown(self):
        self.scheduler.shutdown()
        logger.info("Scheduler shutdown.")

    def schedule_daily_pipeline(self, pipeline_func, hour: int = 9, minute: int = 15):
        """
        Schedules a daily pipeline job (e.g., streaming live data or running the shadow pipeline).
        """
        self.scheduler.add_job(
            pipeline_func,
            "cron",
            day_of_week="mon-fri",
            hour=hour,
            minute=minute,
            id="daily_pipeline",
            replace_existing=True,
        )
        logger.info(f"Scheduled daily pipeline at {hour}:{minute:02d} Mon-Fri.")

    def schedule_weekend_retraining(self, retrain_func):
        """
        Schedules intense model retraining and backtesting on weekends.
        """
        self.scheduler.add_job(
            retrain_func,
            "cron",
            day_of_week="sat",
            hour=2,  # 2 AM Saturday
            id="weekend_retrain",
            replace_existing=True,
        )
        logger.info("Scheduled weekend retraining at 02:00 AM on Saturdays.")
