from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    seoul_api_key: str = ""
    kma_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
    random_seed: int = 42


settings = Settings()
