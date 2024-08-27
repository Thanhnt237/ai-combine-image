const fs = require("fs");
const tf = require("@tensorflow/tfjs-node")
const path = require("node:path");

class Model {
    #STANDARD_SIZE = [224, 224];
    #STANDARD_LAYER_CHANNEL = 3;
    #model = null;

    constructor() {
        this.#model = this.#createModel()
    }

    getModel = () => {
        return this.#model
    }

    loadModel = async () => {
        const pathModel = path.join(__dirname, 'assets/model.json')

        this.#model = await tf.loadLayersModel(`file://${pathModel}`)

        return this.#model
    }

    #createModel = () => {
        const inputLayer = this.#createInputLayer();

        const denseLayer = this.#createDenseLayer(inputLayer);

        const outputLayer = this.#createOutputLayer(denseLayer);

        const model = tf.model({
            inputs: [inputLayer.carInput, inputLayer.personInput],
            outputs: outputLayer
        });

        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        return model
    }

    #createInputLayer = () => {
        const shape = [
            ...this.#STANDARD_SIZE,
            this.#STANDARD_LAYER_CHANNEL
        ]
        return {
            carInput: tf.input({ shape }),
            personInput: tf.input({ shape }),
        }
    }

    #createDenseLayer = (inputLayer) => {
        const carFeatures = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' })
            .apply(inputLayer.carInput);

        const personFeatures = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' })
            .apply(inputLayer.personInput);

        const concatenated = tf.layers.concatenate().apply([carFeatures, personFeatures]);
        const context2d = tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu'}).apply(concatenated)
        const convertedContext2d = tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }).apply(context2d);

        return {
            context_2d: convertedContext2d
        };

    }

    #createOutputLayer = (denseLayer) => {
        return tf.layers.conv2d({ filters: 3, kernelSize: 3, activation: 'sigmoid' }).apply(denseLayer.context_2d);
    }
}

module.exports = Model;