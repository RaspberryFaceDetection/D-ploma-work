import cerberus
import os
import localconfig
import yaml
from attrdict import AttrDict


class Settings:
    """
    Settings class
    """

    CONFIG = None
    CONFIGURATION_FILE_NAME = "conf.ini"
    CONFIGURATION_VALIDATION_FILE_NAME = "config_validation.yml"

    def __init__(
        self,
        file_name=f"{os.path.dirname(os.path.abspath(__file__))}/{CONFIGURATION_FILE_NAME}",
    ):
        """
        Initialize config
        """
        self.read_ini_file(file_name)
        self.validate_ini_file()

    def read_ini_file(self, file_name, config_group="face_recognition"):
        """
        Read data from configuration file

        :param file_name: file name
        :param config_group: config group
        :return:
        """
        with open(file_name, "r") as f:
            config_string = f.read()

        config = localconfig.LocalConfig()
        config.read(config_string)

        params = AttrDict(
            {
                str(key).upper(): value
                for key, value in getattr(config, config_group) or {}
            }
        )

        self.CONFIG = params

    def validate_ini_file(self):
        """
        Validate data from configuration file

        :return:
        """
        try:
            with open(self.CONFIGURATION_VALIDATION_FILE_NAME) as file:
                schema = yaml.load(file.read(), Loader=yaml.Loader)
        except FileNotFoundError:
            raise Exception(f"{self.CONFIGURATION_VALIDATION_FILE_NAME} not found")

        try:
            validator = cerberus.Validator(schema)
            validator.validate(self.CONFIG)
        except cerberus.schema.SchemaError as err:
            raise Exception(str(err))

        if validator.errors:
            raise Exception(validator.errors)


settings = Settings()
