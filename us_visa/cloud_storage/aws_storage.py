import boto3
from us_visa.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union, List
import os
import sys

from us_visa.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from us_visa.exception import USvisaException
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
import pickle


class SimpleStorageService:

    def __init__(self):
        s3_client = S3Client()

        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)

            file_objects = [
                file_object
                for file_object in bucket.objects.filter(Prefix=s3_key)
            ]

            if len(file_objects) > 0:
                return True
            else:
                return False

        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_object(
        object_name: object,
        decode: bool = True,
        make_readable: bool = False
    ) -> Union[StringIO, str]:

        """
        Method Name : read_object

        Description :
            Reads an S3 object.

        Parameters:
            object_name:
                S3 Object resource.

            decode:
                If True, decode the bytes into a string.
                If False, return raw bytes.

            make_readable:
                If True, wrap decoded content inside StringIO.

        Returns:
            StringIO, str, or bytes.

        """

        logging.info(
            "Entered the read_object method of S3Operations class"
        )

        try:

            if decode is True:
                content = object_name.get()["Body"].read().decode()
            else:
                content = object_name.get()["Body"].read()

            if make_readable is True:
                content = StringIO(content)

            logging.info(
                "Exited the read_object method of S3Operations class"
            )

            return content

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:

        """
        Method Name : get_bucket

        Description :
            Gets an S3 bucket object.

        Returns:
            Bucket object.
        """

        logging.info(
            "Entered the get_bucket method of S3Operations class"
        )

        try:

            bucket = self.s3_resource.Bucket(bucket_name)

            logging.info(
                "Exited the get_bucket method of S3Operations class"
            )

            return bucket

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_file_object(
        self,
        filename: str,
        bucket_name: str
    ) -> Union[List[object], object]:

        """
        Method Name : get_file_object

        Description :
            Gets file objects from an S3 bucket using Prefix matching.

        Note:
            This method may return:
                - a single object when exactly one match exists
                - a list when multiple matches exist
        """

        logging.info(
            "Entered the get_file_object method of S3Operations class"
        )

        try:

            bucket = self.get_bucket(bucket_name)

            file_objects = [
                file_object
                for file_object in bucket.objects.filter(Prefix=filename)
            ]

            if len(file_objects) == 1:
                file_objs = file_objects[0]
            else:
                file_objs = file_objects

            logging.info(
                "Exited the get_file_object method of S3Operations class"
            )

            return file_objs

        except Exception as e:
            raise USvisaException(e, sys) from e

    def load_model(
        self,
        model_name: str,
        bucket_name: str,
        model_dir: str = None
    ) -> object:

        """
        Method Name : load_model

        Description :
            Loads a pickle model directly from S3.

        Important:
            Uses an exact S3 object key instead of get_file_object(),
            because get_file_object() uses Prefix matching and can return
            a list when multiple objects match.

        Parameters:
            model_name:
                Name of the model file, e.g. "model.pkl".

            bucket_name:
                S3 bucket name.

            model_dir:
                Optional directory/prefix inside the bucket.

        Returns:
            Deserialized Python model.
        """

        logging.info(
            "Entered the load_model method of S3Operations class"
        )

        try:

            # Build the exact S3 object key.
            if model_dir is None:
                model_file = model_name
            else:
                model_file = f"{model_dir}/{model_name}"

            logging.info(
                f"Loading model from s3://{bucket_name}/{model_file}"
            )

            # IMPORTANT:
            # Do NOT use get_file_object() here.
            #
            # get_file_object() uses:
            #
            # bucket.objects.filter(Prefix=filename)
            #
            # which can return a list.
            #
            # Instead, directly reference the exact S3 object.
            file_object = self.s3_resource.Object(
                bucket_name,
                model_file
            )

            # Read the exact S3 object as bytes.
            model_obj = self.read_object(
                file_object,
                decode=False
            )

            # Deserialize the pickle model.
            model = pickle.loads(model_obj)

            logging.info(
                "Exited the load_model method of S3Operations class"
            )

            return model

        except Exception as e:
            raise USvisaException(e, sys) from e

    def create_folder(
        self,
        folder_name: str,
        bucket_name: str
    ) -> None:

        """
        Method Name : create_folder

        Description :
            Creates a folder-like object in S3.
        """

        logging.info(
            "Entered the create_folder method of S3Operations class"
        )

        try:

            self.s3_resource.Object(
                bucket_name,
                folder_name
            ).load()

        except ClientError as e:

            if e.response["Error"]["Code"] == "404":

                folder_obj = folder_name + "/"

                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=folder_obj
                )

            else:
                pass

        logging.info(
            "Exited the create_folder method of S3Operations class"
        )

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True
    ):

        """
        Method Name : upload_file

        Description :
            Uploads a local file to an S3 bucket.
        """

        logging.info(
            "Entered the upload_file method of S3Operations class"
        )

        try:

            logging.info(
                f"Uploading {from_filename} file to "
                f"{to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename,
                bucket_name,
                to_filename
            )

            logging.info(
                f"Uploaded {from_filename} file to "
                f"{to_filename} file in {bucket_name} bucket"
            )

            if remove is True:

                os.remove(from_filename)

                logging.info(
                    f"Remove is set to {remove}, deleted the file"
                )

            else:

                logging.info(
                    f"Remove is set to {remove}, not deleted the file"
                )

            logging.info(
                "Exited the upload_file method of S3Operations class"
            )

        except Exception as e:
            raise USvisaException(e, sys) from e

    def upload_df_as_csv(
        self,
        data_frame: DataFrame,
        local_filename: str,
        bucket_filename: str,
        bucket_name: str
    ) -> None:

        """
        Method Name : upload_df_as_csv

        Description :
            Uploads a DataFrame as a CSV file to S3.
        """

        logging.info(
            "Entered the upload_df_as_csv method of S3Operations class"
        )

        try:

            data_frame.to_csv(
                local_filename,
                index=None,
                header=True
            )

            self.upload_file(
                local_filename,
                bucket_filename,
                bucket_name
            )

            logging.info(
                "Exited the upload_df_as_csv method of S3Operations class"
            )

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_df_from_object(
        self,
        object_: object
    ) -> DataFrame:

        """
        Method Name : get_df_from_object

        Description :
            Gets a pandas DataFrame from an S3 object.
        """

        logging.info(
            "Entered the get_df_from_object method of S3Operations class"
        )

        try:

            content = self.read_object(
                object_,
                make_readable=True
            )

            df = read_csv(
                content,
                na_values="na"
            )

            logging.info(
                "Exited the get_df_from_object method of S3Operations class"
            )

            return df

        except Exception as e:
            raise USvisaException(e, sys) from e

    def read_csv(
        self,
        filename: str,
        bucket_name: str
    ) -> DataFrame:

        """
        Method Name : read_csv

        Description :
            Reads a CSV file from an S3 bucket.
        """

        logging.info(
            "Entered the read_csv method of S3Operations class"
        )

        try:

            csv_obj = self.get_file_object(
                filename,
                bucket_name
            )

            # If multiple objects match the prefix, fail clearly
            # instead of producing "'list' object has no attribute 'get'".
            if isinstance(csv_obj, list):

                if len(csv_obj) == 0:
                    raise FileNotFoundError(
                        f"No S3 object found with prefix '{filename}' "
                        f"in bucket '{bucket_name}'"
                    )

                raise ValueError(
                    f"Multiple S3 objects found with prefix '{filename}' "
                    f"in bucket '{bucket_name}'. "
                    f"Use an exact S3 key."
                )

            df = self.get_df_from_object(csv_obj)

            logging.info(
                "Exited the read_csv method of S3Operations class"
            )

            return df

        except Exception as e:
            raise USvisaException(e, sys) from e