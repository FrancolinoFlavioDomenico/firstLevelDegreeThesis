import Fastify from "fastify";
import { ethers } from "ethers";
import { createHeliaHTTP } from '@helia/http'
import { unixfs } from '@helia/unixfs'
import { createRequire } from "module";
import { spawn } from 'node:child_process';

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
const helia = await createHeliaHTTP();
const fs = unixfs(helia)

const compiledContract = require("../blockchain/artifacts/src/blockchain/contracts/CheckWeights.sol/CheckWeights.json");
const contractABI = compiledContract.abi;
const contractBytecode = compiledContract.bytecode;
var deployedContractAddress;

function log(msg, type = "debug") {
    console.log(msg)
    fastify.log[type](msg);
}

function catchBlockchainEvent() {
    const contract = new ethers.Contract(deployedContractAddress, contractABI, provider);
    contract.on("WroteWeightsOfRoundForClient", (weights, round, federatedCid, event) => {
        try {
            const args = [weights,round,federatedCid];
            spawn('python', ['src/utils/guard.py',weights,round,federatedCid]);//run python script for weight check
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

//write weights of client {:clientCid} at round {:round} into blockchain
fastify.post("/write/weights/:clientCid/:round", async (request, reply) => {

    try {
        const weights = request.body.weights
        let walletKey = request.body.blockchainCredential

        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const weightAddress = await fs.addBytes(weights)

        const tx = await contract.storeWeightOfRoundForClient(
            weightAddress.toString(),
            request.params.round,
            request.params.clientCid,
        )
        const txResult = await tx.wait()
        reply.send({ txResult: txResult })
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

//write poisoner client {:clientCid}  into blockchain blacklist
fastify.post("/write/blacklist/:clientCid", async (request, reply) => {
    try {
        let walletKey = request.body.blockchainCredential
        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const tx = await contract.addToBlacklist(
            request.params.clientCid,
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

  /* if weigth < avgWeitg of previous round
            add clientCid to blacklist
            write weigt on blockchain
            write blacklist on blockchain
      */

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