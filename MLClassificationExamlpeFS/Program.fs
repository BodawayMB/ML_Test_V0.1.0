// Learn more about F# at http://fsharp.org

open System                    
open Microsoft.ML
open Microsoft.ML.Runtime.Api
open Microsoft.ML.Trainers
open Microsoft.ML.Transforms
open MLClassificationExample   // <= Utilisation des types de l'exemple C#

//[<CLIMutable>]
//type IrisData = {
//    [<Column("0")>]
//    SepalLength:float;

//    [<Column("1")>]
//    SepalWidth:float;

//    [<Column("2")>]
//    PetalLength:float;

//    [<Column("3")>]
//    PetalWidth:float;

//    [<Column("4")>]
//    [<ColumnName("Label")>]
//    Label:string;
//}

//[<CLIMutable>]
//type IrisPrediction = {
//    [<ColumnName("PredictedLabel")>]
//    PredictedLabels:string;
//}

type IrisPrediction() =
    [<ColumnName("PredictedLabel")>]
    member val PredictedLabels = "" with get, set

[<EntryPoint>]
let main argv =

    let pipeline = new LearningPipeline()
    let dataPath = "iris-data.txt"

    let getPredictionConverter = 
        let converter = PredictedLabelColumnOriginalValueConverter()
        converter.PredictedLabelColumn <- "PredictedLabel"
        converter

    let ope: ILearningPipelineItem[] = [|
        TextLoader<IrisData>(dataPath,false,",");
        Dictionarizer("Label");
        ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");
        StochasticDualCoordinateAscentClassifier()
        getPredictionConverter;  
    |]

    ope |> Array.iter pipeline.Add

    let model = pipeline.Train<IrisData, IrisPrediction>()
            
    //var prediction = model.Predict(new IrisData()
    //{
    //    SepalLength = 3.3f,
    //    SepalWidth = 1.6f,
    //    PetalLength = 0.2f,
    //    PetalWidth = 5.1f,
    //})

    Console.WriteLine("Predicted flower type is: {prediction.PredictedLabels}")

    0 // return an integer exit code
