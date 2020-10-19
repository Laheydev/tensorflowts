import React, { SyntheticEvent, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

import "./Main.css";

let linearModel: tf.Sequential;

export default function Main() {
  async function trainNewModel() {
    linearModel = tf.sequential();
    linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    linearModel.compile({ loss: "meanSquaredError", optimizer: "sgd" });

    const xs = tf.tensor1d([3.2, 4.4, 5.5]);
    const ys = tf.tensor1d([1.6, 2.7, 3.5]);

    /* training */
    await linearModel.fit(xs, ys);

    console.log("model trained !");
  }

  function linearPrediction(val: any) {
    const output = linearModel.predict(tf.tensor2d([val], [1, 1])) as tf.Tensor;
    setPrediction(Array.from(output.dataSync())[0]);
  }

  function changeModelData(e: any) {
    setInputValue(e.target.value);
  }

  const [prediction, setPrediction] = useState<any>(undefined);
  const [inputValue, setInputValue] = useState<number>(0);
  useEffect(() => {
    trainNewModel();
    return () => {};
  }, []);

  return (
    <div>
      TensorFlow says {prediction}
      <input
        type="number"
        value={inputValue}
        onChange={(e) => changeModelData(e)}
      />
    </div>
  );
}
