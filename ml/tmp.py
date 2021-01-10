from minio import Minio
from minio.error import S3Error


def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    # client = Minio(
    #     endpoint="play.min.io",
    #     access_key="Q3AM3UQ867SPQQA43P2F",
    #     secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    # )

    # client.list_buckets()
    client = Minio(
        endpoint="0.0.0.0:9001",
        access_key="HFKQGYD2P6DJO4U7N33U",
        secret_key="hO09epo9zT6tslwhh192l0UHbtpCjCYTnE3TYtec",
        secure=False
    )

    # Make 'asiatrip' bucket if not exist.
    found = client.bucket_exists("asiatrip")
    if not found:
        client.make_bucket("asiatrip")
    else:
        print("Bucket 'asiatrip' already exists")

    # # Upload '/home/user/Photos/asiaphotos.zip' as object name
    # # 'asiaphotos-2015.zip' to bucket 'asiatrip'.
    # client.fput_object(
    #     "asiatrip", "asiaphotos-2015.zip", "/home/user/Photos/asiaphotos.zip",
    # )
    # print(
    #     "'/home/user/Photos/asiaphotos.zip' is successfully uploaded as "
    #     "object 'asiaphotos-2015.zip' to bucket 'asiatrip'."
    # )


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)