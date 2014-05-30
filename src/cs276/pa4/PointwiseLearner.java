package cs276.pa4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.pa4.Util;
import util.Pair;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class PointwiseLearner extends Learner {
	
	private LinearRegression model;
	private QueryDocScorer scorer;
	
	public PointwiseLearner() {
		model = new LinearRegression();
		scorer = new QueryDocScorer();
	}

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs, Map<String, Double> additionalFeatures) {
		Map<Query,List<Document>> trainData = null;
		try {
			 trainData = Util.loadTrainData(train_data_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Map<String, Map<String, Double>> relData = null;
		
		try {
			relData = Util.loadRelData(train_rel_file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// create the dataset
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		// go through each query
		for (Query q : trainData.keySet()) {
			String queryStr = q.query;
			
			// go through all documents for that query
			for (Document d : trainData.get(q)) {
				// get the relevance
				double relevance = relData.get(queryStr).get(d.url);
				
				// extract the features from each of the fields in this doc
				Map<String, Double> scores = null;
				try {
					scores = scorer.getSimScore(d, q, idfs);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				double[] instance = scorer.constructFeatureArray(scores, additionalFeatures, relevance);
				Instance inst = new DenseInstance(1.0, instance); 
				dataset.add(inst);
			}
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		System.out.println("COEFFICIENTS: " + this.model.numParameters());
//		for (double coef : this.model.coefficients()) {
//			System.out.println(coef);
//		}
		
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs, Map<String, Double> additionalFeatures) {
		Map<Query,List<Document>> testData = null;
		try {
			testData = Util.loadTrainData(test_data_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// create the dataset
		
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("test_dataset", attributes, 0);
		
		int currIndex = 0;
		
		// go through each query
		for (Query q : testData.keySet()) {
			String queryStr = q.query;
			
			// doc -> index in dataset
			Map<String, Integer> doc2Index = new HashMap<String, Integer>();
			
			// go through all documents for that query
			for (Document d : testData.get(q)) {

				// extract the features from each of the fields in this doc
				Map<String, Double> scores = null;
				try {
					scores = scorer.getSimScore(d, q, idfs);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				double[] instance = scorer.constructFeatureArray(scores, additionalFeatures, -1.0);
				Instance inst = new DenseInstance(1.0, instance); 
				dataset.add(currIndex, inst);
				
				doc2Index.put(d.url, currIndex);
				
				currIndex++;
			}
			
			indexMap.put(queryStr, doc2Index);
		}

		TestFeatures result = new TestFeatures();
		result.features = dataset;
		result.index_map = indexMap;
		
		return result;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		Map<String, List<String>> result = new HashMap<String, List<String>>();
		
		for (String queryStr : tf.index_map.keySet()) {
			List<Pair<String, Double>> docScores = new ArrayList<Pair<String, Double>>();
			
			for (String url : tf.index_map.get(queryStr).keySet()) {
				// get the features for this testing point
				Instance i = tf.features.get(tf.index_map.get(queryStr).get(url));
				
				double predictedRelevance = -1;
				try {
					predictedRelevance = model.classifyInstance(i);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				docScores.add(new Pair(url, predictedRelevance));
			}
			
			// get the docs, sorted by relevance
			List<String> docs = this.getSortedDocs(docScores);
			result.put(queryStr, docs);
		}
		
		return result;
	}

}
