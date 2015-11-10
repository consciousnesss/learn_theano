import os
from boto.s3.connection import S3Connection
import ConfigParser
from StringIO import StringIO
import tempfile
import subprocess


class S3(object):
    '''
    Expects security.properties file in the same folder with the following content:
    AWS_ACCESS_KEY_ID=<YOUR KEY ID>
    AWS_SECRET_KEY=<YOUR KEY>
    '''
    def __init__(self, bucket_name='consciousnesss-storage'):
        security_file = os.path.join(os.path.dirname(__file__), 'security.properties')
        if os.path.exists(security_file):
            # http://stackoverflow.com/questions/9686184/is-there-a-version-of-configparser-that-deals-with-files-with-no-section-headers
            config = ConfigParser.ConfigParser()
            config.readfp(StringIO(u'[DUMMY]\n%s' % open(security_file).read()))
            config = dict(config.items('DUMMY'))
            aws_access_key_id = config['aws_access_key_id']
            aws_secret_key = config['aws_secret_key']
        elif 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_KEY' in os.environ:
            aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
            aws_secret_key = os.environ['AWS_SECRET_KEY']
        else:
            raise Exception("Please put security.properties file in the same folder with this script or define AWS system variables")

        self._connection = S3Connection(aws_access_key_id, aws_secret_key)
        self._bucket = self._connection.get_bucket(bucket_name)

    def cache_folder(self):
        """The absolute path of the base folder for artifacts cache"""
        cache_folder = os.path.expanduser("~/.%s-cache" % self._bucket.name)
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        return cache_folder

    def list_keys(self, prefix=None):
        # http://stackoverflow.com/questions/9954521/s3-boto-list-keys-sometimes-returns-directory-key
        keys = []
        for k in self._bucket.get_all_keys(prefix=prefix):
            if prefix is not None and k.name in [prefix, prefix+'/']:
                continue
            keys.append(k)
        return keys

    def download(self, key_name, download_folder=None, decompress=True):
        if download_folder is None:
            download_folder = self.cache_folder()
        result_filename = os.path.join(download_folder,
                                       os.path.split(key_name)[-1])
        if not os.path.exists(result_filename):
            print("Downloading %s to %s" % (key_name, result_filename))
            k = self._bucket.get_key(key_name)
            k.get_contents_to_filename(result_filename)
        else:
            print("Returning file %s from cache %s" % (key_name, result_filename))
        if decompress:
            return self._try_to_decompress(result_filename)
        else:
            return result_filename

    def _try_to_decompress(self, filename):
        if not filename.endswith('.gz'):
            return filename

        uncompressed_filename=os.path.splitext(filename)[0]
        if not os.path.isfile(uncompressed_filename):
            print("Uncompressing %s to %s" % (filename, uncompressed_filename))
            # gzip dies with MemoryError
            # zlib results in 'incorrect header check'
            # zcat gives an 'out of memory error' (OSX 10.9)
            # use solution from http://stackoverflow.com/questions/13989029/read-in-zlib-file-in-python-results-in-incorrect-header-check
            commands = ['zcat "%s" > "%s"' % (filename, uncompressed_filename),
                        'gzcat "%s" > "%s"' % (filename, uncompressed_filename)]

            for c in commands:
                devnull = open(os.devnull, 'w')
                if subprocess.call(c, shell=True, stdout=devnull, stderr=devnull) == 0:
                    break
            else:
                os.remove(uncompressed_filename)
                raise Exception("Failed to uncompress %s" % (filename,))
        return uncompressed_filename

    def download_folder(self, s3_folder, results_folder=None, decompress=True):
        if results_folder is None:
            results_folder = self.cache_folder()
        assert(os.path.exists(results_folder))
        filenames = []
        for k in self.list_keys(prefix=s3_folder):
            result_filename = os.path.join(results_folder,
                                           os.path.split(k.name)[-1])
            filenames.append(result_filename)
            if not os.path.exists(result_filename):
                print("Downloading %s to %s" % (k.name, result_filename))
                k.get_contents_to_filename(result_filename)
            else:
                print("Returning file %s from cache %s" % (k.name, result_filename))
        if decompress:
            for i, f in enumerate(filenames):
                filenames[i] = self._try_to_decompress(f)
        return filenames


def get_artifact(keyname):
    return S3().download(keyname)
