import { AutoProcessor, RawImage, AutoTokenizer , CLIPTextModelWithProjection,CLIPVisionModelWithProjection } from '@xenova/transformers';
import { Pinecone } from '@pinecone-database/pinecone';

const pinecone = new Pinecone({
    apiKey: 'xxxxx'
});
const index = pinecone.index('imageindex');

const model_id = 'Xenova/clip-vit-base-patch32';

const model = await CLIPVisionModelWithProjection.from_pretrained(model_id);
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const textModel = await CLIPTextModelWithProjection.from_pretrained(model_id);
const  imageProcessor = await AutoProcessor.from_pretrained(model_id);
const  rawImage = await RawImage.read('https://xxxxxxx.s3.ap-northeast-1.amazonaws.com/1000550952.jpg');

const  imageInputs = await imageProcessor(rawImage);

// open jpg file and set to image object　
const {image_embeds} = await model(imageInputs);
const x = Array.from(image_embeds.data);
//const x = image_embeds.data;

const texts = ["building","snow mountain","town in the snow mountain","river","town near a beach","town in europa"];

texts.forEach( (text, index) => {
    textEmbedding(text);
})

function dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
}

function magnitude(a) {
    return Math.sqrt(dotProduct(a, a));
}

function cosineSimilarity(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have the same dimensions");
    }
    
    const magA = magnitude(a);
    const magB = magnitude(b);
    
    if (magA === 0 || magB === 0) {
        throw new Error(
            "Magnitude of one of the vectors is zero, cannot calculate cosine similarity"
        );
    }
    
    return dotProduct(a, b) / (magA * magB);
}


async function textEmbedding(text) {

    const textInputs = tokenizer(text, { padding: true, truncation: true });

    const { text_embeds } = await textModel(textInputs) ;
    const y = Array.from(text_embeds.data);

    const s = cosineSimilarity( x, y );
    console.log(text , s)

    /* query to pinecone */
    const queryResponse = await index.namespace('webcamInfo').query({
        topK: 3,
        vector: y,
    });

    queryResponse.matches.forEach((result) => {
        console.log(text,result.score, result.id);
    })
};
