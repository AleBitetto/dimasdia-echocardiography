# Deploy info

* Use a IAM Role with policies:
  * `AmazonSageMakerFullAccess`
  * Custom with: `s3:ListBucket`, `s3:GetObject`, `s3:DeleteObject`, `s3:PutObject` (for the s3 deploy bucket).
* Current deploy role is: arn:aws:iam::953018779520:role/service-role/AmazonSageMaker-ExecutionRole-20200327T152761`
* Current deploy bucket is `s3://prod.models.sagemaker.defeatcovid19.org`
* Create a deploy file named `model.tar.gz` with content

- model.pth
- code
|- inference.py
|- requirements.txt

* Current model.pth is `AkineticNoDiff_Resnet34_BCE_full_final_state_dict.pth`
* Current deploy folder is `s3://prod.models.sagemaker.defeatcovid19.org/pilot-akinesis-eco-sacco/model.tar.gz`
* Run the deploy notebook (with proper aws profile and cli)


# Invoke Endpoint
Endpoint accepts an "application/dicom" (for files <5MB) and "application/json".

## With Access Token/Secret (Postman)
`POST https://runtime.sagemaker.{{region}}.amazonaws.com/endpoints/{{aws_model_name}}/invocations`

In Authorization
```
Type: AWS Signature
AccessKey: ...
SecretKey: ...
AWS Region: ...
Service Name: sagemaker
```

In Headers
```
Content-Type: 'application/dicom'
```
or
```
Content-Type: 'application/json'
```


In Body
```
Binary with DICOM File
```
or
```
{
  'source': DICOM_URL
}
```
