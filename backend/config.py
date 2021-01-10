import os


class Config(object):
    """Base config, uses staging database server."""
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = "9asdf8980as8df9809sf6a6ds4f3435fa64ˆGggd76HSD57hsˆSDnb"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    IP = os.getenv('IP', '0.0.0.0')
    PORT = os.getenv('PORT', 4444)
    ENV = "development"
    DEVELOPMENT = True
    SECRET_KEY = "secret_for_test_environment"
    OAUTHLIB_INSECURE_TRANSPORT = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True
