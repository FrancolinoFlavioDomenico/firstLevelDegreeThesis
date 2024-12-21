import Fastify from "fastify";
import { ethers } from "ethers";
import { createHeliaHTTP } from '@helia/http'
import { unixfs } from '@helia/unixfs'
import { createRequire } from "module";
import { spawn } from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as crypto from 'node:crypto';

const require = createRequire(import.meta.url);

const fastify = Fastify({
    logger: {
        enabled: true,
        file: "log/node_log.txt",
    },
    bodyLimit: 100 * 1024 * 1024,
});

fastify.register(require('@fastify/multipart'), {
    limits: {
        fileSize: 500 * 1024 * 1024,
    },
    attachFieldsToBody: "keyValues",
});

const provider = new ethers.JsonRpcProvider('http://127.0.0.1:8545/');

const compiledContract = require("../blockchain/artifacts/src/blockchain/contracts/CheckWeights.sol/CheckWeights.json");
const contractABI = compiledContract.abi;
const contractBytecode = compiledContract.bytecode;
var deployedContractAddress;

var datasetName = 'mnist'
var datasetClassNumber = 10

function log(msg, type = "debug") {
    console.log(msg)
    fastify.log[type](msg);
}

function catchBlockchainEvent() {
    const contract = new ethers.Contract(deployedContractAddress, contractABI, provider);
    contract.on("WroteWeightsOfRoundForClient", (weightsHash, round, federatedCid, event) => {
        try {
            const args = [weightsHash, round, federatedCid];
            const child = spawn('python', ['src/utils/guard.py', weightsHash, round, federatedCid,datasetName,datasetClassNumber],{shell:true});//run python script for weight check
            child.stderr.pipe(process.stdout)
        } catch (error) {
            log('Error executing Python script: ' + error, "error");
        }

    }
    )
};

/*********************************************************
 blockchain test route
 **********************************************************/

//----------connection test-----------
fastify.get("/isConnected", async function (request, reply) {
    const isConnected = await !!provider.getNetwork()
    reply.send({ isConnected: isConnected });
});

/*----------------------------------------------------------
                   real used api
----------------------------------------------------------*/
//deploy smart contract on blockchain
fastify.post("/configure/dataset", async function (request, reply) {
    try {
        datasetName = request.body.datasetName;
        datasetClassNumber = request.body.datasetClassNumber;
        reply.send();
    } catch (e) {
        request.log.error({
            e,
            message: "deploy contract failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });
    } finally {
        return reply
    }
});

//deploy smart contract on blockchain
fastify.post("/deploy/contract", async function (request, reply) {
    try {
        const wallet = new ethers.Wallet(request.body.blockchainCredential, provider)
        const contractFactory = new ethers.ContractFactory(contractABI, contractBytecode, wallet);
        const contract = await contractFactory.deploy();
        const deployTransaction = await contract.waitForDeployment();
        deployedContractAddress = deployTransaction.target
        reply.send({ deployTransaction: deployTransaction });
        catchBlockchainEvent();
    } catch (e) {
        request.log.error({
            e,
            message: "deploy contract failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });
    } finally {
        return reply
    }
});


//write weights hash of client {:clientCid} at round {:round} into blockchain
fastify.post("/write/weights/:clientCid/:round", async (request, reply) => {

    try {
        const round = request.params.round;
        const clientCid = request.params.clientCid;

        const weights = request.body.weights
        const walletKey = request.body.blockchainCredential

        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const path = `./data/clientParameters/node/client${clientCid}_round${round}_parameters.pth`
        await fs.writeFile(path,weights,{encoding:"binary"});
        const fileContent = await fs.readFile(path)
        
        const weightHash = crypto.createHash('md5').update(fileContent).digest('hex');

        const tx = await contract.addRoundWeightsReference(
            weightHash,
            round,
            clientCid,
        )
        const txResult = await tx.wait()
        reply.send({ txResult: txResult, weightsHash: weightHash })
    } catch (e) {
        request.log.error({
            e,
            message: "write weight on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});

//write poisoner client {:poisonerCid}  into blockchain blacklist
fastify.post("/write/blacklist/:poisonerCid", async (request, reply) => {
    try {
        let walletKey = request.body.blockchainCredential
        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const tx = await contract.addToBlacklist(
            request.params.poisonerCid,
        )
        const txResult = await tx.wait()

        reply.send({ txResult: txResult })
    } catch (e) {
        request.log.error({
            e,
            message: "write blacklist on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});

//get poisoners blacklist
fastify.get("/blacklist", async (request, reply) => {
    try {
        let walletKey = request.body.blockchainCredential
        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const tx = await contract.getBlacklist(
            request.params.clientCid,
        )
        const txResult = await tx.wait()

        reply.send({ txResult: txResult })
    } catch (e) {
        request.log.error({
            e,
            message: "get blacklist failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});


/*----------------------------------------------------------
              start server
----------------------------------------------------------*/
// Run the server!
fastify.listen({ port: 3000 }, function (err, address) {
    if (err) {
        fastify.log.error(err);
        process.exit(1);
    }
    log(`Server is now listening on ${address}`);
});