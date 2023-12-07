#!/bin/env ruby 

require 'json'
require 'csv'


gold = File.read('data/gold.json')

data = JSON.parse(gold)

o = CSV.open("data/training.tsv", "w:utf-8", col_sep: "\t", quote_char: "\b")

count = 0
data.each do |rec|
  inp = rec['input']
  refs = rec['references']
  id = inp['id']
  cit = inp['reference']
  refs.each do |ref|
      count += 1
      datum = {
        titleName: ref['titleName'],
        volume: ref['volume'],
        partPages: ref['partPages'],
        yearAggr: ref['yearAggr'],
        titleYearStart: ref['titleYearStart'],
        titleYearEnd: ref['titleYearEnd'],
        pageNum: ref['pageNum']
      }
    if ref['isNomenRef']
      score = 0.95
    else
      score = 0.2       
    end
    o << [id, cit.to_json, datum.to_json, score]
  end
end

o.close

