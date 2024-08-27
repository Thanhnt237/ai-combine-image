const Normalizer = require("./normalizer")
const path = require("path");

class Trainer {
    #normalizer = null;

    constructor() {
        this.#normalizer = new Normalizer()
    }

    #loadAssets = () => {
        const inputSize = [224, 224];
        const outputSize = [216, 216];

        const personImage = this.#normalizer.loadImage(path.join(__dirname, 'assets/sample/owner/owner_1.png'), inputSize)
        const carImage = this.#normalizer.loadImage(path.join(__dirname, 'assets/sample/truck/truck_1.png'), inputSize);
        const outputImage = this.#normalizer.loadImage(path.join(__dirname, 'assets/sample/result/result_1.png'), outputSize)

        return {
            personImage,
            carImage,
            outputImage,
        }
    }

    train = (model) => {
        const assets = this.#loadAssets()

        const xTrain = {
            input1: assets.carImage,
            input2: assets.personImage
        };

        const yTrain = assets.outputImage;

        model.fit(xTrain, yTrain, {
            epochs: 10,
            callbacks: {
                onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}`)
            }
        }).then(() => {
            this.saveModel(model)
            return 'completed!'
        }).catch((error) => {
           return error
        });
    }

    saveModel = (model) => {
        const savePath = path.join(__dirname, 'assets')
        model.save(`file://${savePath}`)
    }
}

module.exports = Trainer;