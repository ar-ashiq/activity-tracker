#!/bin/sh

BASE_URL="http://127.0.0.1:8000"

# Server health
curl "$BASE_URL/health"
printf "\n"

# Available activities
curl "$BASE_URL/activities"
printf "\n"

# All consolidated predictions
curl "$BASE_URL/predictions"
printf "\n"

# Predictions for one activity
curl "$BASE_URL/predictions?activity=Sitting"
printf "\n"

# Predictions overlapping a timestamp range
curl "$BASE_URL/predictions?start_timestamp=600&end_timestamp=700"
printf "\n"

# Activity at one timestamp
curl "$BASE_URL/predictions/at?timestamp=600"
printf "\n"

# Plain-English query: activity at a timestamp
curl -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"what was I doing at 600?"}'
printf "\n"

# Plain-English query: total time spent in an activity
curl -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"how long did I spend sitting?"}'
printf "\n"

# Interactive API documentation
printf "Open: $BASE_URL/docs\n"
