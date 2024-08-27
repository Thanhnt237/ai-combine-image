const Model = require("./model");
const Trainer = require("./trainer");
const Predict = require("./predict");
const fs = require("fs");
const path = require("path");
const tf = require('@tensorflow/tfjs-node');

class Main {
    constructor() {

    }

    async start() {
        const model = await new Model().loadModel()
        console.log(model)
        // const trainer = new Trainer()
        // await trainer.train(model)

        const predict = new Predict(model)

        const truck = path.join(__dirname, 'assets/sample/truck/truck_1.png')
        const person = path.join(__dirname, 'assets/sample/owner/owner_1.png')

        await predict.generateImage(truck, person).then(() => console.log('Image generated and saved as output_generated.jpg'))
    }
}

const main = new Main()

main.start().then(() => console.log("uki"))