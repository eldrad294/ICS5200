#
# Module Imports
import uuid
import hashlib
#
class PasswordEncrypter:
    #
    @staticmethod
    def hash_password(password):
        """
        uuid is used to generate a random number
        :param password:
        :return:
        """
        salt = uuid.uuid4().hex
        return hashlib.sha256(salt.encode() + password.encode()).hexdigest() + ':' + salt
    #
    @staticmethod
    def check_password(hashed_password, user_password):
        password, salt = hashed_password.split(':')
        return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()