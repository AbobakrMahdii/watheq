package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct {
	contractapi.Contract
}

type Document struct {
	DocID     string `json:"docId"`
	Hash      string `json:"hash"`
	CID       string `json:"cid"`
	OwnerDID  string `json:"ownerDid"`
	Timestamp string `json:"timestamp"`
	Metadata  string `json:"metadata"`
}

func (s *SmartContract) RecordDocument(ctx contractapi.TransactionContextInterface, docJSON string) error {
	var doc Document

	err := json.Unmarshal([]byte(docJSON), &doc)
	if err != nil {
		return fmt.Errorf("failed to parse document JSON: %v", err)
	}

	if doc.DocID == "" {
		return fmt.Errorf("docId is required")
	}

	return ctx.GetStub().PutState(doc.DocID, []byte(docJSON))
}

func (s *SmartContract) GetDocument(ctx contractapi.TransactionContextInterface, docID string) (*Document, error) {
	if docID == "" {
		return nil, fmt.Errorf("docId is required")
	}

	data, err := ctx.GetStub().GetState(docID)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if data == nil {
		return nil, fmt.Errorf("document %s does not exist", docID)
	}

	var doc Document
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("failed to unmarshal document JSON: %v", err)
	}

	return &doc, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(new(SmartContract))
	if err != nil {
		fmt.Printf("Error create watheq_cc chaincode: %v\n", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting watheq_cc chaincode: %v\n", err)
	}
}
