import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

# ---------------------------------------------------------
# SageMaker Session
# ---------------------------------------------------------

session = sagemaker.Session()

role = "arn:aws:iam::264868026158:role/SageMakerExecutionRole"

# ---------------------------------------------------------
# Pipeline Parameters
# ---------------------------------------------------------

input_data = ParameterString(
    name="InputData"
)

processed_output = ParameterString(
    name="ProcessedOutput"
)

model_output = ParameterString(
    name="ModelOutput"
)

# ---------------------------------------------------------
# Processing Processor
# ---------------------------------------------------------

processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session
)

# ---------------------------------------------------------
# Processing Step
# ---------------------------------------------------------

processing_step = ProcessingStep(
    name="FeatureEngineering",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="features",
            source="/opt/ml/processing/output",
            destination=processed_output
        )
    ],
    code="feature_engineering.py"
)

# ---------------------------------------------------------
# Training Estimator
# ---------------------------------------------------------

estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",
    framework_version="1.2-1",
    py_version="py3",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=model_output,
    sagemaker_session=session
)

# ---------------------------------------------------------
# Training Step
# ---------------------------------------------------------

training_step = TrainingStep(
    name="ModelTraining",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties
            .ProcessingOutputConfig
            .Outputs["features"]
            .S3Output
            .S3Uri
        )
    }
)

# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    name="PredictEC2PatchCompliancePipeline",
    parameters=[
        input_data,
        processed_output,
        model_output
    ],
    steps=[
        processing_step,
        training_step
    ],
    sagemaker_session=session
)

# ---------------------------------------------------------
# Create / Update Pipeline
# ---------------------------------------------------------

pipeline.upsert(
    role_arn=role
)
print("Pipeline deployed successfully.")