package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;

public class Query implements Comparable<Query>{
	String query;
	List<String> words; /* Words with no duplicates and all lower case */
	
	public String[] stopWordsList = { "a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your" };
	public HashSet<String> STOP_WORDS = new HashSet<String>(Arrays.asList(stopWordsList));
			
	private boolean isRemoveStopWords = false;
	
	public Query(String query) {
		
		this.query = new String(query);
		String[] words_array = query.toLowerCase().split(" ");	
		
		if (isRemoveStopWords) {
			// remove stop words from the query
			ArrayList<String> words_list = new ArrayList<String>();
			for (String w : words_array) {
				if (!STOP_WORDS.contains(w)) {
					words_list.add(w);
				}
			}
			
			words_array = new String[words_list.size()];
			for (int i = 0; i < words_list.size(); i++) {
				words_array[i] = words_list.get(i);
			}
		}
		
		// Use LinkedHashSet to remove duplicates
		words_array = (new LinkedHashSet<String>(Arrays.asList(words_array))).toArray(new String[0]);
		this.words = new ArrayList<String>(Arrays.asList(words_array));
	}
	
	@Override
	public int compareTo(Query arg0) {
		return this.query.compareTo(arg0.query);
	}
	
	@Override
	public String toString() {
	  return query;
	}
}
