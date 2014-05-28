package cs276.pa4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import util.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {

	private String DOC_SEPARATOR = "~^~^~^~";
	private LibSVM model;
	private QueryDocScorer scorer;

	// store the feature vector for a given (query, doc) so we don't have to repeatedly compute the same one
	private Map<Pair<String, String>, double[]> vectorCache;
	
	public PairwiseLearner(boolean isLinearKernel){
		try{
			model = new LibSVM();
			scorer = new QueryDocScorer();
		} catch (Exception e){
			e.printStackTrace();
		}

		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}

	public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
		try{
			model = new LibSVM();
		} catch (Exception e){
			e.printStackTrace();
		}

		model.setCost(C);
		model.setGamma(gamma); // only matter for RBF kernel
		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
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

		// store the feature vector for a given (query, doc) so we don't have to repeatedly compute the same one
		vectorCache = new HashMap<Pair<String, String>, double[]>();

		// go through each query
		for (Query q : trainData.keySet()) {
			String queryStr = q.query;

			List<Document> docs = trainData.get(q);

			// go through all pairs of documents for that query
			for (int i = 0; i < docs.size(); i++) {
				for (int j = i + 1; j < docs.size(); j++) {
					Document d1 = trainData.get(q).get(i);
					Document d2 = trainData.get(q).get(j);

					// get the relevance for both docs
					double relevance1 = relData.get(queryStr).get(d1.url);
					double relevance2 = relData.get(queryStr).get(d2.url);

					// skip this pair if they have the same relevance scores
					if (relevance1 == relevance2) {
						continue;
					}

					// the (query, doc) pairs
					Pair currQueryDoc1 = new Pair(queryStr, d1.url);
					Pair currQueryDoc2 = new Pair(queryStr, d2.url);

					// the feature vector for this (query, doc) (see if we cached it already)
					double[] currQueryDocFeatures1 = null;
					double[] currQueryDocFeatures2 = null;

					// check the cache for (query, doc1)
					if (!vectorCache.containsKey(currQueryDoc1)) {
						// extract the features from each of the fields in this doc
						Map<String, Double> scores = null;
						try {
							scores = scorer.getSimScore(d1, q, idfs);
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}

						currQueryDocFeatures1 = scorer.constructFeatureArray(scores, relevance1);
					} else {
						currQueryDocFeatures1 = vectorCache.get(currQueryDoc1);
					}

					// check the cache for (query, doc2)
					if (!vectorCache.containsKey(currQueryDoc2)) {
						// extract the features from each of the fields in this doc
						Map<String, Double> scores = null;
						try {
							scores = scorer.getSimScore(d2, q, idfs);
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}

						currQueryDocFeatures2 = scorer.constructFeatureArray(scores, relevance2);
					} else {
						currQueryDocFeatures2 = vectorCache.get(currQueryDoc2);
					}

					// at this point we have 2 feature vectors: (q, d1) and (q, d2)

					// now need to subtract the vectors
					double[] featureVector = scorer.subtractVectors(currQueryDocFeatures1, currQueryDocFeatures2);

					// find the output label for this pair of vectors (-1 or +1)
					double outputLabel = -100;
					if (relevance1 > relevance2) {
						outputLabel = 1;
					} else if (relevance1 < relevance2) {
						outputLabel = -1;
					}

					// set the relevance of the resulting feature vector
					featureVector[5] = dataset.attribute(5).addStringValue("" + outputLabel);;

					Instance inst = new DenseInstance(1.0, featureVector);
					dataset.add(inst);

				} // end j loop

			} // end i loop

		} // end query loop

		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		// standardize the points between 0 and 1
		Standardize filter = new Standardize();
		Instances newDataset = null;
		try {
			filter.setInputFormat(dataset);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
			newDataset = Filter.useFilter(dataset, filter);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		return newDataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		Map<Query,List<Document>> trainData = null;
		try {
			trainData = Util.loadTrainData(test_data_file);
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
		dataset = new Instances("train_dataset", attributes, 0);

		int currIndex = 0;

		// store the feature vector for a given (query, doc) so we don't have to repeatedly compute the same one
		Map<Pair<String, String>, double[]> vectorCache = new HashMap<Pair<String, String>, double[]>();

		// go through each query
		for (Query q : trainData.keySet()) {
			String queryStr = q.query;

			// doc -> index in dataset
			Map<String, Integer> doc2Index = new HashMap<String, Integer>();

			List<Document> docs = trainData.get(q);

			// go through all pairs of documents for that query
			for (int i = 0; i < docs.size(); i++) {
				for (int j = i + 1; j < docs.size(); j++) {
					Document d1 = trainData.get(q).get(i);
					Document d2 = trainData.get(q).get(j);

					// the (query, doc) pairs
					Pair currQueryDoc1 = new Pair(queryStr, d1.url);
					Pair currQueryDoc2 = new Pair(queryStr, d2.url);

					// the feature vector for this (query, doc) (see if we cached it already)
					double[] currQueryDocFeatures1 = null;
					double[] currQueryDocFeatures2 = null;

					// check the cache for (query, doc1)
					if (!vectorCache.containsKey(currQueryDoc1)) {
						// extract the features from each of the fields in this doc
						Map<String, Double> scores = null;
						try {
							scores = scorer.getSimScore(d1, q, idfs);
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}

						currQueryDocFeatures1 = scorer.constructFeatureArray(scores, -100);
					} else {
						currQueryDocFeatures1 = vectorCache.get(currQueryDoc1);
					}

					// check the cache for (query, doc2)
					if (!vectorCache.containsKey(currQueryDoc2)) {
						// extract the features from each of the fields in this doc
						Map<String, Double> scores = null;
						try {
							scores = scorer.getSimScore(d2, q, idfs);
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}

						currQueryDocFeatures2 = scorer.constructFeatureArray(scores, -100);
					} else {
						currQueryDocFeatures2 = vectorCache.get(currQueryDoc2);
					}

					// at this point we have 2 feature vectors: (q, d1) and (q, d2)

					// now need to subtract the vectors
					double[] featureVector = scorer.subtractVectors(currQueryDocFeatures1, currQueryDocFeatures2);
					Instance inst = new DenseInstance(1.0, featureVector); 
					dataset.add(inst);

					doc2Index.put(d1.url + DOC_SEPARATOR + d2.url, currIndex);

					currIndex++;

				} // end j loop

			} // end i loop

			indexMap.put(queryStr, doc2Index);

		} // end query loop

		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		// standardize the points between 0 and 1
		Standardize filter = new Standardize();
		Instances newDataset = null;
		try {
			filter.setInputFormat(dataset);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
			newDataset = Filter.useFilter(dataset, filter);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		TestFeatures result = new TestFeatures();
		result.features = newDataset;
		result.index_map = indexMap;

		return result;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		double[] weights = ((LibSVM) model).coefficients();
		
		Map<String, List<String>> result = new HashMap<String, List<String>>();

		for (String queryStr : tf.index_map.keySet()) {
			List<Pair<String, Double>> docScores = new ArrayList<Pair<String, Double>>();
			
			HashSet<String> docsAlreadySeen = new HashSet<String>();

			// url1AndUrl2 is a concatenation, with a delimiter, of url for doc1 and url for doc2
			for (String url1AndUrl2 : tf.index_map.get(queryStr).keySet()) {
				String[] urls = url1AndUrl2.split(DOC_SEPARATOR);
				String doc1Url = urls[0];
				String doc2Url = urls[1];
				
				docsAlreadySeen.add(doc1Url);
				docsAlreadySeen.add(doc2Url);
				
				// get the feature vectors for (query, doc1) and (query, doc2)
				double[] featureVector1 = vectorCache.get(new Pair(queryStr, doc1Url));
				double[] featureVector2 = vectorCache.get(new Pair(queryStr, doc2Url));
				
				// if we haven't added scores for these docs, add them
				if (!docsAlreadySeen.contains(doc1Url)) {
					docScores.add(new Pair(doc1Url, scorer.getDotProduct(featureVector1, weights)));
				}
				
				if (!docsAlreadySeen.contains(doc2Url)) {
					docScores.add(new Pair(doc2Url, scorer.getDotProduct(featureVector2, weights)));
				}
				
			}
			
			// get the docs, sorted by relevance
			List<String> docs = this.getSortedDocs(docScores);
			result.put(queryStr, docs);
		}

		return result;
	}

}
