// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
pragma abicoder v2;

contract CheckWeights {
    struct WeightsReferences {
        uint federatedCid;
        mapping(uint => string) weightOfRound;
    }
    WeightsReferences[] public weightsReferences;

    uint[] public blacklist;

    address private owner;

    bool serverCorrupted;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    function getOwner() public view returns (address) {
        return owner;
    }

    event WroteWeightsOfRoundForClient(uint round, uint federatedCid);

    function addRoundWeightsReference(
        string memory _weights,
        uint _round,
        uint _federatedCid
    ) public {
        WeightsReferences storage weightRef;

        if (isClientAlreadyExist(_federatedCid)) {
            weightRef = weightsReferences[findClientIndex(_federatedCid)];
        } else {
            uint lastWeightRef = weightsReferences.length;
            weightsReferences.push();
            weightRef = weightsReferences[lastWeightRef];
            weightRef.federatedCid = _federatedCid;
        }

        weightRef.weightOfRound[_round] = _weights;
        if (_round >= 2 && msg.sender != owner)
            emit WroteWeightsOfRoundForClient(_round, _federatedCid); //throw wrote event for trigger check  of weights

    }

    function getWeightOfRoundOfClient(
        uint _round,
        uint _federatedCid
    ) public view returns (string memory) {
        WeightsReferences storage weightRef = weightsReferences[
            findClientIndex(_federatedCid)
        ];
        return weightRef.weightOfRound[_round];
    }

    function findClientIndex(uint _federatedCid) internal view returns (uint) {
        for (uint i = 0; i < weightsReferences.length; i++) {
            if (weightsReferences[i].federatedCid == _federatedCid) {
                return i;
            }
        }
        revert("Client not found");
    }

    function isClientAlreadyExist(
        uint _federatedCid
    ) internal view returns (bool) {
        for (uint i = 0; i < weightsReferences.length; i++) {
            if (weightsReferences[i].federatedCid == _federatedCid) {
                return true;
            }
        }
        return false;
    }

    function addToBlacklist(uint _federatedCid) public {
        blacklist.push(_federatedCid);
    }

    function setServerCorrupted(bool _serverCorrupted) public {
        serverCorrupted = _serverCorrupted;
    }

    function isPoisonerCid(uint _federatedCid) public view returns (bool) {
        for (uint i = 0; i < blacklist.length; i++) {
            if (blacklist[i] == _federatedCid) {
                return true;
            }
        }
        return false;
    }
}
