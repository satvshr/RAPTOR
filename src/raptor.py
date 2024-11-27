from langsmith import Client as traceable

# @traceable
def raptor_template():
    def raptor(doc_splits):
        return doc_splits
        
    return raptor