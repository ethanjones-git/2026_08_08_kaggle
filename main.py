from processing.kaggle_s3_importer import KaggleS3Importer


class Main:

    def __init__(self):
        pass

    def kaggle_s3_import(self):

        uploader = KaggleS3Importer(bucket_name="2026-08-08-kaggle-bucket")

        uploader.run()

if __name__ == "__main__":

    main = Main()

    main.kaggle_s3_import()