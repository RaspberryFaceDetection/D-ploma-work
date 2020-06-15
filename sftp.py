import pysftp

from settings import settings


class SFTP:
    def __init__(self):
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        self.sftp_connection = pysftp.Connection(host=settings.CONFIG['SFTP_HOST'],
                                                 port=settings.CONFIG['SFTP_PORT'],
                                                 username=settings.CONFIG['SFTP_USER'],
                                                 password=settings.CONFIG['SFTP_PASSWORD'],
                                                 cnopts=cnopts
                                                 )

    def put_file_on_sftp(self, local_file, sftp_file):
        """
        Put local file on sfp

        :param local_file: local file path
        :param sftp_file: remote sftp file path
        :return:
        """
        self.sftp_connection.put(local_file, remotepath=sftp_file)

    def sftp_path_exists(self, sftp_path):
        """
        Check if SFTP path exists

        :param sftp_path: SFTP path to check
        :return:
        """
        return self.sftp_connection.exists(sftp_path)

    def mkdir(self, sftp_dir):
        """
        Makes a sftp dir
        """
        self.sftp_connection.mkdir(sftp_dir)

    def list_dir(self, sftp_dir):
        """
        Get list of directory

        :param sftp_dir:
        :return:
        """
        return self.sftp_connection.listdir(sftp_dir)

    def copy_sftp_to_local(self, sftp_file, local_file):
        """
        Copy file from SFTP to local filesystem

        :param sftp_file:
        :param local_file:
        :return:
        """
        self.sftp_connection.get(sftp_file, local_file)

sftp = SFTP()

