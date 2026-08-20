from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global de la aplicación utilizando pydantic-settings."""

    environment: str = "development"
    log_level: str = "INFO"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "preterm_infant_prediction"

    # API & Service
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
