import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Streamlit setup for vacation preferences
st.title("Autonomous Vacation Planner")

# Collect user preferences
destination = st.text_input("Preferred Destination")
activities = st.text_input("Preferred Activities (e.g., hiking, sightseeing)")
budget = st.number_input("Budget (in USD)", min_value=100, max_value=10000)

if st.button("Plan My Vacation"):
    # Step 1: Generate the vacation plan using Google Gemini and LangChain
    prompt_template = PromptTemplate(
        input_variables=["destination", "activities", "budget"],
        template=(
            "Plan a vacation to {destination}, including {activities}. "
            "Ensure the total cost is under {budget} USD."
        )
    )

    # Create an instance of Google Generative AI (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

    # Create an LLMChain to generate vacation plans
    vacation_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate vacation plan based on user input
    plan = vacation_chain.run({
        "destination": destination,
        "activities": activities,
        "budget": budget
    })

    st.write("Generated Vacation Plan:")
    st.write(plan)

    # Step 2: Summarize the vacation plan
    summary_prompt_template = PromptTemplate(
        input_variables=["plan"],
        template="Summarize this vacation plan: {plan}"
    )

    # Create an LLMChain to summarize the plan
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # Generate the summary
    summary_result = summary_chain.run({"plan": plan})

    st.write("Vacation Summary:")
    st.write(summary_result)

    # Step 3: Confirm and Make Bookings (Optional)
    if st.button("Confirm and Book"):
        st.write("Booking confirmed! You will receive a confirmation email shortly.")
