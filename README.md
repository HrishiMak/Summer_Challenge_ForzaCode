# Summer-Challenge-on-Writer-Verification_TeamForzaCode
This repo compiles the submission files of the Team Forza Code; and includes the trained and inference models, test results and codebase. 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/HrishiMak/Summer-Challenge-on-Writer-Verification_TeamForzaCode.git
   ```
2. Enter current directory
   ```sh
   cd Summer-Challenge-on-Writer-Verification_TeamForzaCode
   ```
3. Install the requirements:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Code

To run the code, you will need to provide the paths to the following files:

### Checkpoints

* The path to the checkpoint for the siamese network:
        ``path_to_checkpoint1: "siamese_model"``
* The path to the checkpoint for the classification network:
        ``path_to_checkpoint2: "classification_model-0.9272"``

### Data

* The path to the test CSV file:
     ``path_to_test_csv: "test.csv"``
* The path to the directory containing the test images:
     ``path_to_test_imgdir: ${path_to_test_imgdir}``

Once you have provided these paths, you can run the code as follows:
 ```sh
      python inference.py --path_to_checkpoint1 ${path_to_checkpoint1} --path_to_checkpoint2 ${path_to_checkpoint2} --path_to_test_csv ${path_to_test_csv} --path_to_test_img ${path_to_test_imgdir}
   ``` 
This will run the inference stage and save the submission file to the current directory.

## Submission File

The submission file will be a CSV file with two columns:

* `id`: The ID of the test pair
* `proba`: The probability that the two images in the test pair were written by the same person.
