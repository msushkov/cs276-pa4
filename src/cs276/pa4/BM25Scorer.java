package cs276.pa4;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BM25Scorer extends Scorer
{
	Map<Query,List<Document>> queryDict;

	///////////////weights///////////////////////////
	double urlweight = 6.0;
	double titleweight = 4.5;
	double bodyweight = 0.1;
	double headerweight = 0.7;
	double anchorweight = 10;

	///////bm25 specific weights///////////////
	double burl = 2;
	double btitle = 1;
	double bheader = 0.5;
	double bbody = 0.75;
	double banchor = 0.1;

	double k1 = 1;
	double pageRankLambda = 1.2;
	double pageRankLambdaPrime = 0.5;
	double pageRankLambdaDubPrime = 4;
	//////////////////////////////////////////

	Map<String, Double> weightParams = new HashMap<String, Double>();

	////////////bm25 data structures--feel free to modify ////////

	Map<Document,Map<String,Double>> lengths;
	Map<String,Double> avgLengths;
	Map<Document,Double> pagerankScores;

	//////////////////////////////////////////

	public BM25Scorer(Map<String,Double> idfs, Map<Query,List<Document>> queryDict)
	{
		this.idfs = idfs;
		this.queryDict = queryDict;
		this.calcAverageLengths();

		weightParams.put("url", urlweight);
		weightParams.put("title", titleweight);
		weightParams.put("body", bodyweight);
		weightParams.put("header", headerweight);
		weightParams.put("anchor", anchorweight);

		weightParams.put("burl", burl);
		weightParams.put("btitle", btitle);
		weightParams.put("bheader", bheader);
		weightParams.put("bbody", bbody);
		weightParams.put("banchor", banchor);
	}

	// queryDict is a map of query -> (url -> document)
	//sets up average lengths for bm25, also handles pagerank
	public void calcAverageLengths()
	{
		lengths = new HashMap<Document,Map<String,Double>>();
		avgLengths = new HashMap<String,Double>();
		pagerankScores = new HashMap<Document,Double>();

		//normalize avgLengths
		for (Query q : queryDict.keySet()) {
			for (Document curr : queryDict.get(q)) {

				// pagerank
				if (!pagerankScores.containsKey(curr)) {
					pagerankScores.put(curr, Math.log10(curr.page_rank));
				}

				Map<String, Double> currDoc = new HashMap<String, Double>();

				for (String tfType : this.TFTYPES) {
					if (!avgLengths.containsKey(tfType)) {
						avgLengths.put(tfType, 0.0);
					}

					int currLength = 0;

					if (tfType.equals("url") && curr.url != null) {
						currLength = getNumUrlTokens(curr.url);

					} else if (tfType.equals("title") && curr.title != null) {
						currLength = getNumTokens(curr.title);

					} else if (tfType.equals("header") && curr.headers != null) {
						currLength = 0;
						for (String h : curr.headers) {
							currLength += getNumTokens(h);
						}

					} else if (tfType.equals("body")) {
						currLength = curr.body_length;

					} else if (tfType.equals("anchor") && curr.anchors != null) {
						currLength = 0;
						for (String anchor : curr.anchors.keySet()) {
							currLength += getNumTokens(anchor) * curr.anchors.get(anchor);
						}
					}

					avgLengths.put(tfType, avgLengths.get(tfType) + currLength);

					currDoc.put(tfType, (double) currLength);
				}

				lengths.put(curr, currDoc);
			}
		}

		// have the sums for each type; now compute the avg using # docs
		for (String type : this.TFTYPES) {
			avgLengths.put(type, avgLengths.get(type) / lengths.size());
		}

	}


	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,
			Document d, Map<String,Double> idfs) {
		double score = 0.0;

		// for each term in the query
		for (String term : tfQuery.keySet()) {
			double w = 0.0;

			for (String type : tfs.keySet()) {
				w += weightParams.get(type) * tfs.get(type).get(term);
			}

			double idfComponent = -1;
			if (idfs.containsKey(term)) {
				idfComponent = idfs.get(term);
			} else {
				idfComponent = Math.log10(idfs.size() + NUM_DOCS);
			}

			score += w * idfComponent / (k1 + w);
		}

		return score;// + getPageRankScore(d);
	}

	private double getPageRankScore(Document d) {
		// tune the pagerank function
		// double funcVal = pagerankScores.get(d) / (pageRankLambdaPrime + pagerankScores.get(d));
		//double funcVal = Math.log10(pageRankLambdaPrime + pagerankScores.get(d));
		double funcVal = 1 / (pageRankLambdaPrime + 
				Math.exp(-1.0 * pageRankLambdaDubPrime * pagerankScores.get(d)));

		return pageRankLambda * funcVal;
	}

	//do bm25 normalization
	public void normalizeTFs(Map<String, Map<String, Double>> tfs, Document d, Query q)
	{
		for (String type : tfs.keySet()) {
			for (String term : tfs.get(type).keySet()) {
				if (avgLengths.get(type) == 0) {
					tfs.get(type).put(term, 0.0);
				} else {
					double newVal = tfs.get(type).get(term);

					double x = (lengths.get(d).get(type) / avgLengths.get(type)) - 1.0;
					double Bf = weightParams.get("b" + type);

					double veryNewVal = newVal / (1.0 + Bf * x);

					tfs.get(type).put(term, veryNewVal);

				}
			}
		}
	}

	public double getBM25Score(Document d, Query q, Map<String, Double> idfs) throws Exception { 
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		this.normalizeTFs(tfs, d, q);

		Map<String,Double> tfQuery = getQueryFreqs(q);

		return getNetScore(tfs, q, tfQuery, d, idfs);
	}

	private int getNumTokens(String str) {
		str = str.replaceAll("[^A-Za-z0-9 ]", "");
		return str.split(" ").length;
	}

	private int getNumUrlTokens(String str) {
		return str.split("[^A-Za-z0-9 ]").length;
	}

}
