const tf = require("@tensorflow/tfjs-node")
const fs = require("fs");
const path = require("path");
const Normalizer = require("./normalizer");

class Predict {
    #normalizer = null;
    #model = null;

    constructor(model) {
        this.#model = model
        this.#normalizer = new Normalizer()
    }

    generateImage = async (carImage, personImage) => {
        const carTensor = this.#normalizer.loadImage(carImage);
        const personTensor = this.#normalizer.loadImage(personImage);

        const output = this.#model.predict([carTensor, personTensor]);

        const outputImage = output.squeeze();

        // Lưu ảnh kết quả ra file
        const buffer = await tf.node.encodeJpeg(outputImage);
        fs.writeFileSync(path.join(__dirname, 'output_generated.jpg'), buffer);
    }
}

module.exports = Predict;