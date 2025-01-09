from predict import Predictor
import time

def main():
    # Create an instance of the Predictor
    predictor = Predictor()

    # Call the setup method to initialize the model
    predictor.setup(first_block_cache=True)

    # Define default arguments
    default_args = {
        "prompt": "A cat walks on the grass, realistic",
        "width": 848,
        "height": 480,
        "num_frames": 129,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "fps": 15,
        "seed": 0
    }

    # Number of times to call predict
    n = 1  # You can change this to any number of iterations you want

    # Call the predict method n times with default arguments
    for i in range(n):
        t1 = time.time()
        output_path = predictor.predict(**default_args)
        t2 = time.time()
        print(f"Prediction {i+1} output saved to: {output_path}")
        print(f"Time taken: {t2 - t1} seconds")

if __name__ == "__main__":
    main()