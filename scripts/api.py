from fastapi import FastAPI
import torch
import numpy as np
from src.models.molgan import Generator
from src.utils.data_processing import graph_to_smiles
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the RL-trained Generator model
generator = Generator()
generator.load_state_dict(torch.load('models/generator_rl_final.pth'))
generator.eval()

class MoleculeRequest(BaseModel):
    num_molecules: int

class MoleculeResponse(BaseModel):
    smiles: List[str]

@app.post("/generate_molecules", response_model=MoleculeResponse)
def generate_molecules(request: MoleculeRequest):
    z = torch.randn(request.num_molecules, 128)
    generated_data = generator(z)
    
    generated_smiles = []
    for data in generated_data:
        adj_matrix = np.round(data[:64]).astype(int)  # Adjust based on actual size
        features = data[64:].astype(int)
        smiles = graph_to_smiles(adj_matrix, features)
        if smiles:
            generated_smiles.append(smiles)
    
    return MoleculeResponse(smiles=generated_smiles)

# Run the API with: uvicorn api:app --reload
