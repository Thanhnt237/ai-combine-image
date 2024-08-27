const tf = require("@tensorflow/tfjs-node")
const fs = require("fs")

class Normalizer {
    #STANDARD_SIZE = [224, 224];

    loadImage = (filePath, size = this.#STANDARD_SIZE) => {
        const imageBuffer = fs.readFileSync(filePath);
        const imageTensor = tf.node.decodeImage(imageBuffer, 3);
        const resizedImage = tf.image.resizeBilinear(imageTensor, size);
        return resizedImage.expandDims(0);
    }
}

module.exports = Normalizer;