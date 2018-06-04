using System;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class TaxiTrip
    {
        [Column(ordinal: "0")]
        public string vendor_id;
        [Column(ordinal: "1")]
        public string rate_code;
        [Column(ordinal: "2")]
        public float passenger_count;
        [Column(ordinal: "3")]
        public float trip_time_in_secs;
        [Column(ordinal: "4")]
        public float trip_distance;
        [Column(ordinal: "5")]
        public string payment_type;
        [Column(ordinal: "6")]
        public float fare_amount;

    }

    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float fare_amount;
    }

    class Program
    {
        const string DataPath = @"C:\Users\Michel\source\repos\ConsoleApp2\ConsoleApp1\taxi-fare-test.csv";
        const string TestDataPath = @"C:\Users\Michel\source\repos\ConsoleApp2\ConsoleApp1\taxi-fare-train.csv";

        static void Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = Train();
            Evaluate(model);
        }

        public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","));
            pipeline.Add(new ColumnCopier(("fare_amount", "Label")));
            pipeline.Add(new CategoricalOneHotVectorizer("vendor_id",
                                             "rate_code",
                                             "payment_type"));
            pipeline.Add(new ColumnConcatenator("Features",
                                    "vendor_id",
                                    "rate_code",
                                    "passenger_count",
                                    "trip_distance",
                                    "payment_type"));

            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader<TaxiTrip>(TestDataPath, useHeader: true, separator: ",");
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine("Rms=" + metrics.Rms);
            Console.WriteLine("RSquared = " + metrics.RSquared);
        }

    }
}
