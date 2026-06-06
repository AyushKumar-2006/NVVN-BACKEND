from django.apps import AppConfig


class PowerConfig(AppConfig):
    name = 'power'

    def ready(self):
        # Auto-start the MERIT CG poller when running under a real server
        # process (runserver child / gunicorn). Guarded inside should_autostart()
        # so it never runs during migrations, shells, or the poll subprocess.
        # Failures here must never block app startup.
        try:
            from power import scheduler
            if scheduler.should_autostart():
                scheduler.start()
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("MeritScheduler").warning(
                f"scheduler not started: {exc}"
            )





