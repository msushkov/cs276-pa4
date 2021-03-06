package cs276.pa4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class Learning2Rank {
	
	// TODO: change the kernel type here!
	// false is RBF
	private static boolean isLinearKernel = false;
	
	private static Map<String, Double> additionalFeatures = new HashMap<String, Double>();

	
	private static void addFeatures() {
		// TODO: add the features here!
		//additionalFeatures.put("bm25", 1.0);
		//additionalFeatures.put("smallestwindow", 1.0);
		//additionalFeatures.put("pagerank", 1.0);
		//additionalFeatures.put("percentage_of_query_terms_in_body", 1.0);
		//additionalFeatures.put("percentage_of_query_terms_in_anchors", 1.0);
		//additionalFeatures.put("num_of_unique_anchors", 1.0);
		//additionalFeatures.put("title_length", 1.0);
	}

	// overloaded method to accept C and gamma as params
	public static Classifier train(String train_data_file, String train_rel_file, int task, Map<String,Double> idfs, double C, double gamma) {
		//System.err.println("## Training with feature_file =" + train_data_file + ", rel_file = " + train_rel_file + " ... \n");
		Classifier model = null;
		Learner learner = null;

		if (task == 1) {
			learner = new PointwiseLearner();
		} else if (task == 2) {
			learner = new PairwiseLearner(C, gamma, isLinearKernel);
		} else if (task == 3) {
			learner = new PairwiseLearner(C, gamma, false);
		} else if (task == 4) {

			/* 
			 * @TODO: Your code here, extra credit 
			 * */
			System.err.println("Extra credit");

		}

		/* Step (1): construct your feature matrix here */
		Instances data = learner.extract_train_features(train_data_file, train_rel_file, idfs, additionalFeatures);

		/* Step (2): implement your learning algorithm here */
		model = learner.training(data);

		return model;
	}

	// overloaded method to accept C and gamma
	public static Map<String, List<String>> test(String test_data_file, Classifier model, int task,
			Map<String,Double> idfs, double C, double gamma){
		System.err.println("## Testing with feature_file=" + test_data_file + " ... \n");
		Map<String, List<String>> ranked_queries = new HashMap<String, List<String>>();
		Learner learner = null;
		
		if (task == 1) {
			learner = new PointwiseLearner();
		} else if (task == 2) {
			learner = new PairwiseLearner(isLinearKernel);
		} else if (task == 3) {
			learner = new PairwiseLearner(C, gamma, false);
		} else if (task == 4) {

			/* 
			 * @TODO: Your code here, extra credit 
			 * */
			System.err.println("Extra credit");

		}

		/* Step (1): construct your test feature matrix here */
		TestFeatures tf = learner.extract_test_features(test_data_file, idfs, additionalFeatures);

		/* Step (2): implement your prediction and ranking code here */
		ranked_queries = learner.testing(tf, model);

		return ranked_queries;
	}



	/* This function output the ranking results in expected format */
	public static void writeRankedResultsToFile(Map<String,List<String>> ranked_queries, PrintStream ps) {
		for (String query : ranked_queries.keySet()){
			ps.println("query: " + query.toString());

			for (String url : ranked_queries.get(query)) {
				ps.println("  url: " + url);
			}
		}
	}


	public static void main(String[] args) throws IOException {
		if (args.length != 4 && args.length != 5) {
			System.err.println("Input arguments: " + Arrays.toString(args));
			System.err.println("Usage: <train_data_file> <train_data_file> <test_data_file> <task> [ranked_out_file]");
			System.err.println("  ranked_out_file (optional): output results are written into the specified file. "
					+ "If not, output to stdout.");
			return;
		}

		String train_data_file = args[0];
		String train_rel_file = args[1];
		String test_data_file = args[2];
		int task = Integer.parseInt(args[3]);
		String ranked_out_file = "";
		if (args.length == 5){
			ranked_out_file = args[4];
		}

		/* Populate idfs */
		String dfFile = "df.txt";
		Map<String,Double> idfs = null;
		try {
			idfs = Util.loadDFs(dfFile);
		} catch(IOException e){
			e.printStackTrace();
		}
		
		addFeatures();

		// grid search over parameters
		// double[] Cs = { Math.pow(2, -3), 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 };
		// double[] gammas = { Math.pow(2, -7), Math.pow(2, -6), Math.pow(2, -5), Math.pow(2, -4), Math.pow(2, -3), Math.pow(2, -2), Math.pow(2, -1) };

		// use the optimal parameters discovered through grid search
		double[] Cs = { 0.125 };
		double[] gammas = { 0.03125 };

		for (double C : Cs) {
			for (double gamma : gammas) {
				System.out.println("C: " + C);
				System.out.println("gamma: " + gamma);

				/* Train & test */
				System.err.println("### Running task" + task + "...");		
				Classifier model = train(train_data_file, train_rel_file, task, idfs, C, gamma);

				/* performance on the training data */
				Map<String, List<String>> trained_ranked_queries = test(train_data_file, model, task, idfs, C, gamma);
				String trainOutFile="tmp.train.ranked";
				writeRankedResultsToFile(trained_ranked_queries, new PrintStream(new FileOutputStream(trainOutFile)));

				NdcgMain ndcg = new NdcgMain(train_rel_file);
				System.err.println("# Trained NDCG=" + ndcg.score(trainOutFile));

				(new File(trainOutFile)).delete();

				Map<String, List<String>> ranked_queries = test(test_data_file, model, task, idfs, C, gamma);

				/* Output results */
				if (ranked_out_file.equals("")){ /* output to stdout */
					writeRankedResultsToFile(ranked_queries, System.out);
				} else { 						/* output to file */
					try {
						writeRankedResultsToFile(ranked_queries, new PrintStream(new FileOutputStream(ranked_out_file)));
					} catch (FileNotFoundException e) {
						e.printStackTrace();
					}

				} // end if

				NdcgMain ndcg1 = new NdcgMain("data/pa4.rel.dev");
				double ndcgScore = ndcg1.score(ranked_out_file);
				System.out.println("SCORE: " + ndcgScore);

			} // end loop over gamma

		} // end loop over C

	}
}
