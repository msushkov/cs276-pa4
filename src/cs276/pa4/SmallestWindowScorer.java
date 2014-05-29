package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import util.Pair;

//doesn't necessarily have to use task 2 (could use task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead)
public class SmallestWindowScorer extends QueryDocScorer
{
	/////smallest window specific hyperparameters////////
	
	double B = 2;    	    

	//////////////////////////////

	// query -> document url -> smallest window
	private Map<Query, Map<Document, Double>> smallestWindowBoost = new HashMap<Query, Map<Document, Double>>();

	
	public SmallestWindowScorer(Map<String, Double> idfs, Map<Query,List<Document>> queryDict) {
		this.idfs = idfs;
		initSmallestWindows(queryDict);
	}

	// queryDict: query -> (url -> Document)
	public void initSmallestWindows(Map<Query,List<Document>> queryDict) {
		// for each query, compute the smallest window for each of the zones of each document
		for (Query q : queryDict.keySet()) {
			smallestWindowBoost.put(q, new HashMap<Document, Double>());

			// for each document
			for (Document currDoc : queryDict.get(q)) {		

				double val = -1;
				
				// find the smallest window for this q, d pair
				int smallestWindow = findSmallestWindow(q, currDoc);
				
				if (smallestWindow == Integer.MAX_VALUE) {
					val = 1.0;
				} else {
					// find # of unique terms in the query
					Set<String> uniqueQueryTerms = new HashSet<String>();
					String[] tokens = q.query.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "").split(" ");
					for (String term : tokens) {
						uniqueQueryTerms.add(term);
					}
					
					// if the query has 1 unique term, then it should be boosted to the max
					if (uniqueQueryTerms.size() == 1 || smallestWindow == uniqueQueryTerms.size()) {
						val = B;
					} else {
						// if the window is larger than the query size
						val = 1.0 + (B - 1) * Math.exp(-(smallestWindow - uniqueQueryTerms.size()));
					}
				}

				smallestWindowBoost.get(q).put(currDoc, val);
			}
		}
	}


	private int findSmallestWindow(Query q, Document currDoc) {
		int smallestWindow = Integer.MAX_VALUE;

		// for each field type
		for (String type : this.TFTYPES) {			
			if (type.equals("title") && currDoc.title != null) {
				int currWindow = getWindow(convertToMap(q, currDoc.title));
				
				if (currWindow < smallestWindow) {
					smallestWindow = currWindow;
				}

			} else if (type.equals("body") && currDoc.body_hits != null) {
				String[] tokens = q.query.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "").split(" ");
				
				// the tokens in the content string
				String body = "";
				for (String s : currDoc.body_hits.keySet()) {
					body += s;
					body += " ";
				}
				String[] contents = body.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "").split(" ");
				
				// make sure very term in the query is contained in the body
				if (everyTermOccurs(tokens, contents)) {
					//System.out.println("every term occurs in the body");
					
					int currWindow = getWindow(currDoc.body_hits);

					if (currWindow < smallestWindow) {
						smallestWindow = currWindow;
					}
				}
				
			} else if (type.equals("header") && currDoc.headers != null) {
				for (String header : currDoc.headers) {
					int currWindow = getWindow(convertToMap(q, header));

					if (currWindow < smallestWindow) {
						smallestWindow = currWindow;
					}
				}

			} else if (type.equals("anchor") && currDoc.anchors != null) {
				for (String anchor : currDoc.anchors.keySet()) {
					int currWindow = getWindow(convertToMap(q, anchor));

					if (currWindow < smallestWindow) {
						smallestWindow = currWindow;
					}
				}
			}
		}

		return smallestWindow;
	}

	private Map<String, List<Integer>> convertToMap(Query q, String contentStr) {
		Map<String, List<Integer>> termIndices = 
				new HashMap<String, List<Integer>>();
		
		String[] tokens = q.query.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "").split(" ");
		
		// the tokens in the content string
		String[] contents = contentStr.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "").split(" ");
		
		if (everyTermOccurs(tokens, contents)) {
			for (String term : tokens) {
				// indices for current term
				List<Integer> indices = new ArrayList<Integer>();
				
				// get the indices of the query tokens in the title
				for (int i = 0; i < contents.length; i++) {
					if (term.equals(contents[i])) {
						indices.add(i);
					}
				}
			
				termIndices.put(term, indices);
			}
		}
		
		return termIndices;
	}

	private int getWindow(Map<String, List<Integer>> termIndices) {
		// infinity
		if (termIndices.isEmpty()) {
			return Integer.MAX_VALUE;
		}

		// create a mapping of term -> position in list of indices
		Map<String, Integer> positions = new HashMap<String, Integer>();
		
		// initialize to all 0's
		for (String term : termIndices.keySet()) {
			positions.put(term, 0);
		}
		
		
		int bestWindowSize = Integer.MAX_VALUE;
		boolean canContinue = true;
		
		// go until we reach the end of one of the lists
		while (canContinue) {
			Pair<Integer, String> minValAndToken = getLocalWindowSize(termIndices, positions);
			int currWindowSize = minValAndToken.getFirst();
			String minToken = minValAndToken.getSecond();
			
			if (currWindowSize < bestWindowSize) {
				bestWindowSize = currWindowSize;
			}
			
			canContinue = incrementIndex(positions, termIndices, minToken);
		}
		
		return bestWindowSize;
	}
	
	// Returns the min value as well as the token that gave that value
	private Pair<Integer, String> getLocalWindowSize(Map<String, List<Integer>> termIndices, Map<String, Integer> positions) {
		// find the min and the max; take the difference
		
		int min = Integer.MAX_VALUE;
		int max = -1 * Integer.MAX_VALUE;
		String minToken = "";
		
		for (String key : termIndices.keySet()) {
			int currVal = termIndices.get(key).get(positions.get(key));
			
			if (currVal < min) {
				min = currVal;
				minToken = key;
			}
			
			if (currVal > max) {
				max = currVal;
			}
		}
		
		return new Pair((max - min), minToken);
	}
	
	// Returns false if we have reached the end of the list at key.
	private boolean incrementIndex(Map<String, Integer> positions, Map<String, List<Integer>> termIndices, String key) {
		int currIndex = positions.get(key);
		int len = termIndices.get(key).size();
		
		// cant increment index any more - reached the end of the index list for this term
		if (currIndex >= len - 1) {
			return false;
		} 
		
		positions.put(key, positions.get(key) + 1);
		return true;
	}

	// Return true if every term in terms occurs in contents.
	private boolean everyTermOccurs(String[] terms, Object[] contents) {
		Set<String> cont = new HashSet<String>();
		for (Object c : contents) {
			cont.add((String) c);
		}

		for (String t : terms) {
			if (!cont.contains(t)) {
				return false;
			}
		}

		return true;
	}


	/*
	 * Returns the boost, which is a nonlinear function of the smallest window.
	 */
	public double getSmallestWindow(Document d, Query q, Map<String, Double> idfs) throws Exception {
		return smallestWindowBoost.get(q).get(d);
	}

}
